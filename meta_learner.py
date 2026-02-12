"""
Meta-Learning System: Trade Reflection & Adaptation

This module implements a "learning from mistakes" layer that:
1. Analyzes historical trade results to find patterns in losses.
2. Creates adjustment factors based on conditions (e.g., RSI, volatility).
3. Applies these adjustments to future predictions WITHOUT changing the core strategy.

The core strategy (OLS model on technical + sentiment features) remains unchanged.
This layer only re-weights the final rankings based on learned patterns.
"""
import os
import pandas as pd
import numpy as np
import yaml
import json
import hashlib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetaLearner:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, 'config.yaml')
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        base_dir = os.path.dirname(os.path.abspath(config_path))
        self.results_dir = self.config.get('output', {}).get('results_dir', './results')
        if not os.path.isabs(self.results_dir):
            self.results_dir = os.path.join(base_dir, self.results_dir)
            
        self.feature_store_dir = os.path.join(base_dir, 'feature_store')
        # Keep runtime state in data/ so it's not committed and can be cached in CI.
        data_dir = os.path.join(base_dir, "data")
        self.meta_state_file = os.path.join(data_dir, 'meta_learner_state.json')
        db_rel = self.config.get('data', {}).get('cache_path', './data/trading_bot.db')
        self.db_path = db_rel if os.path.isabs(db_rel) else os.path.join(base_dir, db_rel)

        meta_cfg = self.config.get("meta_learning", {})
        self.cooldown_days_after_exit = int(meta_cfg.get("cooldown_days_after_exit", 5))
        # 0.0 => block, 0.2 => heavily downrank, 1.0 => no effect
        self.cooldown_penalty = float(meta_cfg.get("cooldown_penalty", 0.0))
        self.min_trades_for_penalty = int(meta_cfg.get("min_trades_for_penalty", 3))
        self.win_rate_threshold = float(meta_cfg.get("win_rate_threshold", 0.40))
        
        # Load or initialize state
        self.state = self._load_state()

    @staticmethod
    def _digest_penalties(penalties: dict) -> str:
        """
        Stable signature of current penalties to avoid repeating identical insight blocks.
        """
        try:
            payload = json.dumps(penalties or {}, sort_keys=True, separators=(",", ":"))
        except Exception:
            payload = str(penalties or {})
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _load_state(self):
        # Ensure data dir exists.
        try:
            os.makedirs(os.path.dirname(self.meta_state_file), exist_ok=True)
        except Exception:
            pass

        # Migration: older versions stored state in repo root. Copy once.
        legacy_state = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta_learner_state.json")
        if (not os.path.exists(self.meta_state_file)) and os.path.exists(legacy_state):
            try:
                with open(legacy_state, "r") as src:
                    payload = src.read()
                with open(self.meta_state_file, "w") as dst:
                    dst.write(payload)
            except Exception:
                pass

        if os.path.exists(self.meta_state_file):
            with open(self.meta_state_file, 'r') as f:
                return json.load(f)
        return {
            'symbol_penalties': {},  # Per-symbol penalty based on loss streak
            'symbol_penalty_details': {},  # Optional per-symbol stats used in reports
            'condition_rules': [],   # Learned rules: e.g., "if RSI > 70 and lost, reduce confidence"
            'penalties_digest': None,
            'penalties_changed': True,
            'last_updated': None
        }

    def _save_state(self):
        self.state['last_updated'] = datetime.now().isoformat()
        try:
            os.makedirs(os.path.dirname(self.meta_state_file), exist_ok=True)
        except Exception:
            pass
        with open(self.meta_state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        logger.info(f"Meta-learner state saved to {self.meta_state_file}")

    def _load_trades_from_results(self, lookback_days: int) -> pd.DataFrame:
        if not os.path.exists(self.results_dir):
            return pd.DataFrame()
        trade_files = sorted([f for f in os.listdir(self.results_dir) if f.startswith('trades_') and f.endswith('.csv')])
        if not trade_files:
            return pd.DataFrame()
        recent_files = trade_files[-lookback_days:]
        frames = []
        for fname in recent_files:
            try:
                df = pd.read_csv(os.path.join(self.results_dir, fname))
            except Exception:
                continue
            if df.empty:
                continue
            if "symbol" not in df.columns:
                continue
            if "strat_return" not in df.columns and "realized_pnl" in df.columns:
                df = df.rename(columns={"realized_pnl": "strat_return"})
            if "strat_return" not in df.columns:
                continue
            frames.append(df[["symbol", "strat_return"]].copy())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_trades_from_db(self, lookback_days: int) -> pd.DataFrame:
        if not os.path.exists(self.db_path):
            return pd.DataFrame()
        import sqlite3
        cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)).date().isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql(
                "SELECT symbol, realized_pnl as strat_return, exit_date FROM positions WHERE status='CLOSED' AND exit_date >= ?",
                conn,
                params=(cutoff,),
            )
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()
        if df.empty:
            return pd.DataFrame()
        return df[["symbol", "strat_return"]].copy()

    def get_exit_cooldown_symbols(self, as_of_date=None) -> set:
        """Symbols exited recently (cooldown window). Used to avoid immediate re-entry."""
        if self.cooldown_days_after_exit <= 0:
            return set()
        if not os.path.exists(self.db_path):
            return set()
        import sqlite3
        if as_of_date is None:
            as_of = pd.Timestamp.utcnow().date()
        else:
            as_of = pd.to_datetime(as_of_date).date()
        cutoff = (pd.Timestamp(as_of) - pd.Timedelta(days=self.cooldown_days_after_exit)).date().isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM positions WHERE status='CLOSED' AND exit_date >= ?",
                (cutoff,),
            ).fetchall()
        except Exception:
            rows = []
        finally:
            conn.close()
        return {r[0] for r in rows if r and r[0]}

    def analyze_past_trades(self, lookback_days=30):
        """
        Analyze past trade results to find patterns in losses.
        Updates internal state with penalty factors and condition rules.
        """
        trades_df = self._load_trades_from_results(int(lookback_days))
        if trades_df.empty:
            trades_df = self._load_trades_from_db(int(lookback_days))

        if trades_df.empty:
            # If there are no recent trades, clear penalties so we don't repeat stale insights forever.
            if self.state.get("symbol_penalties"):
                logger.info("No recent trades found; clearing stale meta-learning penalties.")
                self.state["symbol_penalties"] = {}
                self.state["symbol_penalty_details"] = {}
                self.state["penalties_changed"] = True
                self.state["penalties_digest"] = self._digest_penalties({})
                self._save_state()
            else:
                logger.warning("Not enough trade history for meta-learning. Skipping.")
            return

        # ---- Analysis 1: Per-Symbol Loss Streak ----
        symbol_stats = trades_df.groupby('symbol').agg(
            total_trades=('strat_return', 'count'),
            wins=('strat_return', lambda x: (x > 0).sum()),
            avg_return=('strat_return', 'mean')
        )
        symbol_stats['win_rate'] = symbol_stats['wins'] / symbol_stats['total_trades']

        prev_digest = self.state.get("penalties_digest")
        new_penalties = {}
        new_details = {}

        # Penalize symbols with low win rate
        for symbol, row in symbol_stats.iterrows():
            if row['win_rate'] < self.win_rate_threshold and row['total_trades'] >= self.min_trades_for_penalty:
                # 0.0..1.0 scale: lower win rate => stronger penalty (bounded by 0.5 here)
                penalty = 0.5 + (row['win_rate'] / max(1e-9, self.win_rate_threshold)) * 0.5  # 0.5..1.0
                new_penalties[symbol] = float(penalty)
                new_details[symbol] = {
                    "total_trades": int(row["total_trades"]),
                    "wins": int(row["wins"]),
                    "win_rate": float(row["win_rate"]),
                    "avg_return": float(row["avg_return"]),
                }
                logger.info(f"Applied penalty {penalty:.2f} to {symbol} (win_rate: {row['win_rate']:.2%})")

        # Replace penalties atomically so we can compute "changed" cleanly.
        self.state["symbol_penalties"] = new_penalties
        self.state["symbol_penalty_details"] = new_details
        new_digest = self._digest_penalties(new_penalties)
        self.state["penalties_changed"] = (prev_digest != new_digest)
        self.state["penalties_digest"] = new_digest

        # ---- Analysis 2: Feature-Condition Rules ----
        # For each losing trade, check if we can correlate with feature values
        # This requires joining trades with features (simplified here)
        
        losing_trades = trades_df[trades_df['strat_return'] < 0]
        if len(losing_trades) > 5:
            # Placeholder: In a full implementation, we'd join with feature values
            # and find correlations (e.g., losses when RSI > 70)
            logger.info(f"Analyzed {len(losing_trades)} losing trades for pattern detection.")
        
        self._save_state()
        return symbol_stats

    def get_confidence_adjustments(self, rankings_df):
        """
        Apply learned adjustments to the strategy's rankings.
        Returns a modified rankings DataFrame with 'adjusted_score' column.
        """
        rankings = rankings_df.copy()
        
        # Apply symbol penalties
        rankings['penalty'] = rankings['symbol'].map(lambda s: self.state['symbol_penalties'].get(s, 1.0))

        # Cooldown after exit to avoid recycling the same names
        cooldown_syms = self.get_exit_cooldown_symbols()
        rankings['cooldown'] = rankings['symbol'].isin(cooldown_syms)
        rankings['cooldown_penalty'] = np.where(rankings['cooldown'], self.cooldown_penalty, 1.0)
        rankings['penalty'] = rankings['penalty'] * rankings['cooldown_penalty']
        
        # Adjusted score = original prediction * penalty
        rankings['adjusted_score'] = rankings['predicted_return'] * rankings['penalty']
        
        # Re-rank based on adjusted score
        rankings = rankings.sort_values('adjusted_score', ascending=False)
        
        return rankings

    def get_daily_insights(self):
        """Return a summary of what the Meta-Learner is currently applying."""
        insights = []

        cooldown_syms = self.get_exit_cooldown_symbols()
        if cooldown_syms and self.cooldown_days_after_exit > 0:
            shown = sorted(list(cooldown_syms))[:8]
            if self.cooldown_penalty == 0.0:
                insights.append(f"Cooldown active (no re-entry for {self.cooldown_days_after_exit}d after exit): {', '.join(shown)}")
            else:
                insights.append(f"Cooldown active ({self.cooldown_days_after_exit}d, penalty={self.cooldown_penalty:.2f}): {', '.join(shown)}")
            if len(cooldown_syms) > len(shown):
                insights.append(f"  - ...and {len(cooldown_syms) - len(shown)} others.")
        
        # 1. Penalties
        penalties = self.state.get('symbol_penalties', {})
        if penalties:
            changed = bool(self.state.get("penalties_changed", True))
            details = self.state.get("symbol_penalty_details", {}) or {}
            if not changed:
                # Avoid repeating identical blocks every run/email.
                shown = list(penalties.items())[:3]
                syms = ", ".join([s for s, _ in shown])
                insights.append(f"Underperforming assets unchanged since last run ({len(penalties)} active): {syms}")
            else:
                insights.append(f"Analyzing {len(penalties)} underperforming assets:")
                for sym, penalty in list(penalties.items())[:5]:  # Show top 5
                    d = details.get(sym) if isinstance(details, dict) else None
                    if isinstance(d, dict) and d.get("total_trades"):
                        wr = float(d.get("win_rate", 0.0))
                        tt = int(d.get("total_trades", 0))
                        insights.append(
                            f"  - {sym}: penalty={penalty:.2f} (win_rate={wr:.0%}, trades={tt})"
                        )
                    else:
                        insights.append(f"  - {sym}: Confidence reduced by {(1-penalty):.0%} due to low win rate.")
                if len(penalties) > 5:
                    insights.append(f"  - ...and {len(penalties)-5} others.")
        else:
            insights.append("No specific symbol penalties active (all systems nominal).")
            
        # 2. General Rules (Future placeholder)
        rules = self.state.get('condition_rules', [])
        if rules:
            insights.append(f"Active General Pattern Rules: {len(rules)}")
            
        return "\n".join(insights)

if __name__ == "__main__":
    meta = MetaLearner()
    stats = meta.analyze_past_trades()
    if stats is not None:
        print("\n=== Symbol Performance ===")
        print(stats)
        print("\n=== Current Penalties ===")
        print(meta.state['symbol_penalties'])
