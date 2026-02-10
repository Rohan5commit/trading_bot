import os
import shutil
import logging
import sqlite3
from datetime import datetime
import yaml
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')

def reset_database():
    """Archive the current database and start fresh"""
    if not os.path.exists(CONFIG_PATH):
        logger.error("config.yaml not found!")
        return

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve DB Path
    db_rel_path = config['data']['cache_path']
    if not os.path.isabs(db_rel_path):
        db_path = os.path.join(BASE_DIR, db_rel_path)
    else:
        db_path = db_rel_path

    if not os.path.exists(db_path):
        logger.info("No database to reset.")
        return

    # Archive
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_name = f"trading_bot_archive_{timestamp}.db"
    archive_path = os.path.join(os.path.dirname(db_path), archive_name)
    
    try:
        shutil.move(db_path, archive_path)
        logger.info(f"Database Reset Complete.")
        logger.info(f"Old DB archived at: {archive_path}")
        logger.info("The bot will create a fresh, empty database on the next run.")
        
        # Also clean results folder if desired, but maybe keep for reference
        # results_dir = os.path.join(BASE_DIR, 'results')
        # ... logic to archive results ...
        
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")

def _resolve_db_path(config):
    db_rel_path = config['data']['cache_path']
    if not os.path.isabs(db_rel_path):
        return os.path.join(BASE_DIR, db_rel_path)
    return db_rel_path

def _get_latest_price(conn, symbol):
    row = conn.execute(
        "SELECT date, close FROM prices WHERE symbol=? ORDER BY date DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    if not row:
        return None, None
    return row[0], float(row[1])

def reset_pnl_keep_positions(tables=("positions", "positions_ai")):
    """
    Reset BOTH realized + unrealized P&L while keeping open positions:
    - Closed trades are archived (status set to ARCHIVED) so totals go back to zero.
    - Open trades have their cost basis reset to the latest close (unrealized -> 0),
      and TP is recalculated off the new basis (side-aware).
    """
    if not os.path.exists(CONFIG_PATH):
        logger.error("config.yaml not found!")
        return False

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    db_path = _resolve_db_path(config)
    if not os.path.exists(db_path):
        logger.error("Database not found: %s", db_path)
        return False

    tp_pct = float(((config.get("trading") or {}).get("take_profit_pct", 0.03)) or 0.03)

    conn = sqlite3.connect(db_path)
    try:
        for table in tables:
            # Archive closed positions so realized totals reset to zero.
            conn.execute(f"UPDATE {table} SET status='ARCHIVED' WHERE status='CLOSED'")

            # Reset open positions cost basis to latest close so unrealized resets to ~0.
            rows = conn.execute(
                f"SELECT id, symbol, side FROM {table} WHERE status='OPEN'"
            ).fetchall()
            for pid, sym, side in rows:
                side_u = str(side or "LONG").upper()
                date, close = _get_latest_price(conn, sym)
                if close is None:
                    continue
                if side_u == "SHORT":
                    target = close * (1 - tp_pct)
                else:
                    target = close * (1 + tp_pct)
                conn.execute(
                    f"""
                    UPDATE {table}
                    SET entry_date=?, entry_price=?, target_price=?
                    WHERE id=?
                    """,
                    (date, close, target, pid),
                )

        conn.commit()
    finally:
        conn.close()

    logger.info("P&L reset complete for tables: %s", ", ".join(tables))
    return True

def reset_all_state(
    tables=("positions", "positions_ai"),
    reset_positions=True,
    reset_meta_learner=True,
    reset_results_reports=True,
    vacuum_sqlite=False,
):
    """
    Full "fresh start" without deleting price history:
    - Deletes ALL rows in positions tables (OPEN + CLOSED) so P&L and open positions reset to zero.
    - Resets meta_learner_state.json so penalties/insights don't repeat from past runs.
    - Removes prior report artifacts in ./results (daily_report/unrealized/trades) but keeps email markers
      so the scheduler doesn't resend historical emails.
    """
    if not os.path.exists(CONFIG_PATH):
        logger.error("config.yaml not found!")
        return False

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}

    db_path = _resolve_db_path(config)
    if not os.path.exists(db_path):
        logger.error("Database not found: %s", db_path)
        return False

    conn = sqlite3.connect(db_path)
    try:
        if reset_positions:
            for table in tables:
                try:
                    conn.execute(f"DELETE FROM {table}")
                    logger.info("Cleared table: %s", table)
                except Exception as exc:
                    logger.warning("Failed clearing table %s: %s", table, exc)
        if vacuum_sqlite:
            try:
                conn.execute("VACUUM")
                logger.info("SQLite VACUUM complete.")
            except Exception as exc:
                logger.warning("SQLite VACUUM failed: %s", exc)
        conn.commit()
    finally:
        conn.close()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if reset_meta_learner:
        meta_path = os.path.join(base_dir, "meta_learner_state.json")
        try:
            if os.path.exists(meta_path):
                os.remove(meta_path)
            logger.info("Meta-learner state reset: %s", meta_path)
        except Exception as exc:
            logger.warning("Failed to reset meta-learner state: %s", exc)

    if reset_results_reports:
        results_dir = os.path.join(base_dir, "results")
        if os.path.isdir(results_dir):
            patterns = [
                os.path.join(results_dir, "daily_report_*.csv"),
                os.path.join(results_dir, "daily_report_ai_*.csv"),
                os.path.join(results_dir, "unrealized_*.csv"),
                os.path.join(results_dir, "unrealized_ai_*.csv"),
                os.path.join(results_dir, "trades_*.csv"),
            ]
            removed = 0
            for pat in patterns:
                for path in glob.glob(pat):
                    try:
                        os.remove(path)
                        removed += 1
                    except Exception:
                        pass
            logger.info("Removed %s report artifact files from results/ (kept email_sent_*.ok).", removed)

    logger.info("Full state reset complete (positions + reports + meta).")
    return True

if __name__ == "__main__":
    import sys
    mode = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
    if mode in {"pnl", "pnl_keep_positions", "keep_positions"}:
        confirm = input("Reset realized+unrealized P&L for BOTH accounts but KEEP positions open? (yes/no): ")
        if confirm.lower() == "yes":
            ok = reset_pnl_keep_positions()
            print("P&L reset complete." if ok else "P&L reset failed.")
        else:
            print("Reset cancelled.")
    elif mode in {"reset_state", "state", "fresh"}:
        confirm = input("RESET positions + report artifacts + meta-learner state (keeps price history)? (yes/no): ")
        if confirm.lower() == "yes":
            ok = reset_all_state()
            print("State reset complete." if ok else "State reset failed.")
        else:
            print("Reset cancelled.")
    else:
        confirm = input("Are you sure you want to RESET all performance history (archive DB)? (yes/no): ")
        if confirm.lower() == 'yes':
            reset_database()
            print("Database reset complete.")
        else:
            print("Reset cancelled.")
