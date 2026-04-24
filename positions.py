"""
Position Tracker - Hold Until Take Profit

This module manages open positions:
1. Opens positions when signals are generated.
2. Holds positions indefinitely until Take Profit is hit.
3. Tracks unrealized P&L for open positions.
4. Records realized P&L when positions are closed.
"""
import pandas as pd
import numpy as np
import yaml
import logging
import os
import sqlite3
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PositionTracker:
    def __init__(self, config_path=None, table_name="positions"):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, 'config.yaml')
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Resolve DB path relative to config.yaml so scheduler runs (with varying cwd)
        # always read/write the same SQLite file.
        base_dir = os.path.dirname(os.path.abspath(config_path))
        db_rel = self.config['data']['cache_path']
        self.db_path = db_rel if os.path.isabs(db_rel) else os.path.join(base_dir, db_rel)
        self.tp_pct = self.config['trading'].get('take_profit_pct', 0.03)
        self.table_name = self._validate_table_name(table_name)
        self._init_tables()

    def _target_price_for_side(self, entry_price, side):
        side = str(side or "LONG").upper()
        if side == "SHORT":
            return float(entry_price) * (1 - self.tp_pct)
        return float(entry_price) * (1 + self.tp_pct)

    @staticmethod
    def _validate_table_name(name: str) -> str:
        # Prevent SQL injection via table_name.
        safe = str(name or "").strip()
        if not safe:
            raise ValueError("table_name is required")
        for ch in safe:
            if not (ch.isalnum() or ch == "_"):
                raise ValueError(f"Invalid table_name: {name}")
        return safe

    def _init_tables(self):
        """Create positions table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Open positions table
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT DEFAULT 'LONG',
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                target_price REAL NOT NULL,
                status TEXT DEFAULT 'OPEN',
                exit_date TEXT,
                exit_price REAL,
                realized_pnl REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Lightweight migrations for older DBs
        cols = [row[1] for row in cursor.execute(f"PRAGMA table_info({self.table_name})").fetchall()]
        if "side" not in cols:
            cursor.execute(f"ALTER TABLE {self.table_name} ADD COLUMN side TEXT DEFAULT 'LONG'")
            cursor.execute(f"UPDATE {self.table_name} SET side='LONG' WHERE side IS NULL")
        
        conn.commit()
        conn.close()
        logger.info("Position tables initialized.")

    def open_position(self, symbol, entry_date, entry_price, quantity, side="LONG"):
        """Open a new position"""
        side = str(side or "LONG").upper()
        if side not in {"LONG", "SHORT"}:
            raise ValueError(f"Unsupported side: {side}")

        target_price = self._target_price_for_side(entry_price, side)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if we already have an open position for this symbol
        existing = cursor.execute(
            f"SELECT id FROM {self.table_name} WHERE symbol=? AND status='OPEN'",
            (symbol,)
        ).fetchone()
        
        if existing:
            logger.info(f"Already have open position for {symbol}, skipping.")
            conn.close()
            return None
        
        cursor.execute(
            f"""
            INSERT INTO {self.table_name} (symbol, side, entry_date, entry_price, quantity, target_price, status)
            VALUES (?, ?, ?, ?, ?, ?, 'OPEN')
            """,
            (symbol, side, entry_date, entry_price, quantity, target_price)
        )
        
        position_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Opened position #{position_id}: {symbol} {side} @ {entry_price:.2f}, TP @ {target_price:.2f}")
        return position_id

    def add_to_position(self, symbol, add_date, add_price, quantity, side=None):
        """Add capital to an existing open position and blend the average entry price."""
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return None

        try:
            add_price = float(add_price or 0.0)
            quantity = float(quantity or 0.0)
        except (TypeError, ValueError):
            return None

        if add_price <= 0.0 or quantity <= 0.0:
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        existing = cursor.execute(
            f"""
            SELECT id, side, entry_date, entry_price, quantity
            FROM {self.table_name}
            WHERE symbol=? AND status='OPEN'
            """,
            (symbol,),
        ).fetchone()

        if not existing:
            conn.close()
            return None

        pos_id, existing_side, existing_entry_date, existing_entry_price, existing_qty = existing
        existing_side = str(existing_side or "LONG").upper()
        if side is not None and str(side or "LONG").upper() != existing_side:
            conn.close()
            raise ValueError(f"Side mismatch for add_to_position({symbol})")

        existing_entry_price = float(existing_entry_price or 0.0)
        existing_qty = float(existing_qty or 0.0)
        new_qty = existing_qty + quantity
        if new_qty <= 0.0:
            conn.close()
            return None

        blended_entry = ((existing_entry_price * existing_qty) + (add_price * quantity)) / new_qty
        target_price = self._target_price_for_side(blended_entry, existing_side)

        cursor.execute(
            f"""
            UPDATE {self.table_name}
            SET entry_price=?, quantity=?, target_price=?
            WHERE id=?
            """,
            (blended_entry, new_qty, target_price, pos_id),
        )
        conn.commit()
        conn.close()

        logger.info(
            "Added to position #%s: %s %s +%.4f @ %.2f, new avg %.2f, TP @ %.2f",
            pos_id,
            symbol,
            existing_side,
            quantity,
            add_price,
            blended_entry,
            target_price,
        )
        return {
            "id": pos_id,
            "symbol": symbol,
            "side": existing_side,
            "entry_date": existing_entry_date or add_date,
            "entry_price": blended_entry,
            "quantity": new_qty,
            "added_quantity": quantity,
            "target_price": target_price,
        }

    def check_and_close_positions(self, check_date=None):
        """
        Check all open positions against the day's price data.
        Close positions that hit their Take Profit target.
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get all open positions
        open_positions = pd.read_sql(
            f"SELECT * FROM {self.table_name} WHERE status='OPEN'", conn
        )
        
        if open_positions.empty:
            logger.info("No open positions to check.")
            conn.close()
            return []
        
        closed = []
        cursor = conn.cursor()
        
        for _, pos in open_positions.iterrows():
            symbol = pos['symbol']
            target_price = pos['target_price']
            entry_price = pos['entry_price']
            side = str(pos.get('side', 'LONG') or 'LONG').upper()
            
            # Get today's price data
            if check_date:
                date_filter = f"AND date='{check_date}'"
            else:
                date_filter = ""
            
            price_data = pd.read_sql(
                f"SELECT * FROM prices WHERE symbol='{symbol}' {date_filter} ORDER BY date DESC LIMIT 1",
                conn
            )
            
            if price_data.empty:
                continue
                
            latest = price_data.iloc[0]
            high_price = latest['high']
            close_price = latest['close']
            low_price = latest['low']
            check_date_actual = latest['date']
            
            tp_hit = False
            if side == "LONG":
                tp_hit = high_price >= target_price
            elif side == "SHORT":
                tp_hit = low_price <= target_price

            if tp_hit:
                exit_price = target_price
                if side == "LONG":
                    realized_pnl = (exit_price - entry_price) / entry_price
                    realized_pnl_dollars = (exit_price - entry_price) * pos['quantity']
                    reason = f"Take profit hit at {target_price:.2f} (+{self.tp_pct:.1%})"
                else:
                    realized_pnl = (entry_price - exit_price) / entry_price
                    realized_pnl_dollars = (entry_price - exit_price) * pos['quantity']
                    reason = f"Take profit hit at {target_price:.2f} (-{self.tp_pct:.1%})"
                
                cursor.execute(
                    f"""
                    UPDATE {self.table_name}
                    SET status='CLOSED', exit_date=?, exit_price=?, realized_pnl=?
                    WHERE id=?
                    """,
                    (check_date_actual, exit_price, realized_pnl, pos['id'])
                )
                
                closed.append({
                    'symbol': symbol,
                    'side': side,
                    'entry_date': pos['entry_date'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'realized_pnl': realized_pnl,
                    'realized_pnl_dollars': realized_pnl_dollars,
                    'quantity': pos['quantity'],
                    'exit_date': check_date_actual,
                    'target_price': target_price,
                    'reason': reason
                })
                logger.info(f"CLOSED {symbol} {side} @ {exit_price:.2f} (TP HIT) - P&L: {realized_pnl:.2%}")
        
        conn.commit()
        conn.close()
        
        return closed

    def get_unrealized_pnl(self):
        """Calculate unrealized P&L for all open positions"""
        conn = sqlite3.connect(self.db_path)
        
        open_positions = pd.read_sql(
            f"SELECT * FROM {self.table_name} WHERE status='OPEN'", conn
        )
        
        if open_positions.empty:
            conn.close()
            return pd.DataFrame()
        
        results = []
        for _, pos in open_positions.iterrows():
            symbol = pos['symbol']
            entry_price = pos['entry_price']
            side = str(pos.get('side', 'LONG') or 'LONG').upper()
            
            # Get latest close price
            latest = pd.read_sql(
                "SELECT close, date FROM prices WHERE symbol=? ORDER BY date DESC LIMIT 1",
                conn,
                params=(symbol,),
            )
            
            if latest.empty:
                continue
                
            current_price = latest.iloc[0]['close']
            current_price_date = latest.iloc[0]['date']
            if side == "LONG":
                unrealized_pnl = (current_price - entry_price) / entry_price
                unrealized_pnl_dollars = (current_price - entry_price) * pos['quantity']
            else:
                unrealized_pnl = (entry_price - current_price) / entry_price
                unrealized_pnl_dollars = (entry_price - current_price) * pos['quantity']
            
            results.append({
                'symbol': symbol,
                'side': side,
                'entry_date': pos['entry_date'],
                'entry_price': entry_price,
                'quantity': pos['quantity'],
                'current_price': current_price,
                'current_price_date': current_price_date,
                'target_price': pos['target_price'],
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_dollars': unrealized_pnl_dollars,
                'distance_to_tp': abs(pos['target_price'] - current_price) / current_price if current_price else None
            })
        
        conn.close()
        return pd.DataFrame(results)

    def get_open_positions(self):
        """Return all open positions"""
        conn = sqlite3.connect(self.db_path)
        open_positions = pd.read_sql(
            f"SELECT * FROM {self.table_name} WHERE status='OPEN'", conn
        )
        conn.close()
        return open_positions

    def get_portfolio_summary(self):
        """Get summary of all positions (open and closed)"""
        conn = sqlite3.connect(self.db_path)
        
        open_count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {self.table_name} WHERE status='OPEN'", conn).iloc[0]['cnt']
        closed = pd.read_sql(f"SELECT * FROM {self.table_name} WHERE status='CLOSED'", conn)
        
        conn.close()
        
        total_realized = closed['realized_pnl'].sum() if not closed.empty else 0
        total_realized_dollars = 0.0
        if not closed.empty:
            # Dollar P&L must be side-aware for shorts.
            side = closed.get("side")
            if side is None:
                side = "LONG"
            side = side.fillna("LONG").astype(str).str.upper()
            long_mask = side.eq("LONG")
            short_mask = side.eq("SHORT")

            pnl_long = ((closed['exit_price'] - closed['entry_price']) * closed['quantity']).where(long_mask, 0.0)
            pnl_short = ((closed['entry_price'] - closed['exit_price']) * closed['quantity']).where(short_mask, 0.0)
            total_realized_dollars = float((pnl_long.sum() + pnl_short.sum()) or 0.0)
        win_rate = (closed['realized_pnl'] > 0).mean() if not closed.empty else 0
        
        unrealized_df = self.get_unrealized_pnl()
        total_unrealized = unrealized_df['unrealized_pnl'].sum() if not unrealized_df.empty else 0
        total_unrealized_dollars = (
            unrealized_df['unrealized_pnl_dollars'].sum() if not unrealized_df.empty else 0.0
        )
        
        return {
            'open_positions': open_count,
            'closed_positions': len(closed),
            'total_realized_pnl': total_realized,
            'total_realized_pnl_dollars': total_realized_dollars,
            'total_unrealized_pnl': total_unrealized,
            'total_unrealized_pnl_dollars': total_unrealized_dollars,
            'win_rate': win_rate
        }

if __name__ == "__main__":
    tracker = PositionTracker()
    
    # Show current state
    print("\n=== Portfolio Summary ===")
    summary = tracker.get_portfolio_summary()
    for k, v in summary.items():
        print(f"{k}: {v}")
    
    print("\n=== Open Positions (Unrealized P&L) ===")
    unrealized = tracker.get_unrealized_pnl()
    if not unrealized.empty:
        print(unrealized.to_string())
    else:
        print("No open positions.")
