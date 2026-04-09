import json
from dotenv import load_dotenv
import logging
from utils import get_sgt_now
import os
import random
import sqlite3
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml
from utils import get_sgt_now

from ingest_prices import PriceIngestor
from ingest_news import NewsIngestor
from features import FeatureEngineer
from train import ModelManager
from positions import PositionTracker
from portfolio import PortfolioManager
from backtest_signals import build_signal_snapshot
from state_recovery import recover_positions_from_seed, enforce_position_cap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _get_config_path(config_path=None):
    if config_path:
        return config_path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'config.yaml')


def _load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _resolve_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(base_dir, path_value)


def _record_pipeline_issue(pipeline_stats, severity, source, message, max_items=20):
    """Record pipeline warnings/errors in a bounded structure that can be emailed."""
    if not isinstance(pipeline_stats, dict):
        return
    sev = str(severity or "ERROR").strip().upper()
    if sev not in {"ERROR", "WARNING"}:
        sev = "ERROR"

    key = "error_count" if sev == "ERROR" else "warning_count"
    pipeline_stats[key] = int(pipeline_stats.get(key, 0) or 0) + 1

    issues = pipeline_stats.get("issues")
    if not isinstance(issues, list):
        issues = []
        pipeline_stats["issues"] = issues

    issue = {
        "time": get_sgt_now().strftime("%H:%M:%S"),
        "severity": sev,
        "source": str(source or "pipeline"),
        "message": str(message or ""),
    }
    if len(issues) < int(max_items):
        issues.append(issue)
    else:
        pipeline_stats["issue_overflow_count"] = int(pipeline_stats.get("issue_overflow_count", 0) or 0) + 1


def _finalize_pipeline_health(pipeline_stats):
    if not isinstance(pipeline_stats, dict):
        return
    errors = int(pipeline_stats.get("error_count", 0) or 0)
    warnings = int(pipeline_stats.get("warning_count", 0) or 0)
    failed = int(pipeline_stats.get("tickers_failed", 0) or 0)
    if failed > errors:
        errors = failed
    pipeline_stats["error_count"] = errors
    pipeline_stats["warning_count"] = warnings
    pipeline_stats["run_health"] = "ERROR" if errors > 0 else ("WARNING" if warnings > 0 else "OK")


class DailyBacktester:
    def __init__(self, config_path=None):
        self.config_path = _get_config_path(config_path)
        self.config = _load_config(self.config_path)
        self.base_dir = os.path.dirname(os.path.abspath(self.config_path))
        self.db_path = _resolve_path(self.base_dir, self.config['data']['cache_path'])
        self.feature_store_dir = os.path.join(self.base_dir, 'feature_store')
        self.registry_path = os.path.join(self.base_dir, 'model_registry.json')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.results_dir = _resolve_path(
            self.base_dir,
            self.config.get('output', {}).get('results_dir', './results')
        )
        self.universe_path = _resolve_path(self.base_dir, self.config['universe']['source'])
        self.core_tracker = PositionTracker(self.config_path, table_name="positions")
        self.ai_tracker = PositionTracker(self.config_path, table_name="positions_ai")
        self.feature_engineer = FeatureEngineer(self.config_path)
        self.init_all_tables()

    def init_all_tables(self):
        """Ensure all required database tables exist before any processing starts."""
        from ingest_prices import PriceIngestor
        from ingest_news import NewsIngestor
        from positions import PositionTracker

        # Initialize core tables
        PriceIngestor(self.config_path).init_db()
        NewsIngestor(self.config_path).init_db()
        PositionTracker(self.config_path, table_name="positions")
        PositionTracker(self.config_path, table_name="positions_ai")
        logger.info("All database tables initialized successfully.")

    def get_prediction_for_date(self, symbol, signal_date):
        """Get prediction using features as of signal_date (T-1)."""
        if not os.path.exists(self.registry_path):
            return None

        with open(self.registry_path, 'r') as f:
            registry = json.load(f)

        if symbol not in registry:
            return None

        reg_entry = registry[symbol]
        model_id = reg_entry['latest_model']
        model_path = os.path.join(self.models_dir, f"{model_id}.npy")

        if not os.path.exists(model_path):
            return None

        coeffs = np.load(model_path)
        features_list = reg_entry['features']

        # Compute features on-the-fly from SQLite (minimizes disk usage by avoiding feature_store CSVs).
        df = self.feature_engineer.generate(symbol)
        if df is None or df.empty:
            return None
        if "date" not in df.columns:
            return None

        df["date"] = pd.to_datetime(df["date"])
        row = df[df["date"] == pd.to_datetime(signal_date)]
        if row.empty:
            return None

        # Ensure all required feature columns exist.
        for col in features_list:
            if col not in row.columns:
                return None

        X = row[features_list].values[0]
        X_bias = np.insert(X, 0, 1.0)
        pred_return = X_bias @ coeffs

        return pred_return

    def get_predictions_for_date_bulk(self, symbols, signal_date, conn):
        """
        Faster S&P500-scale predictor:
        - Loads recent prices for all symbols in one query
        - Computes technical features via groupby
        - Uses per-symbol OLS coefficients from models/ and model_registry.json
        """
        if not os.path.exists(self.registry_path):
            return []

        with open(self.registry_path, "r") as f:
            registry = json.load(f) or {}

        # Only symbols with an existing model are eligible.
        eligible = []
        for s in symbols:
            sym = str(s or "").strip().upper()
            if not sym:
                continue
            reg = registry.get(sym)
            if not isinstance(reg, dict):
                continue
            model_id = reg.get("latest_model")
            if not model_id:
                continue
            model_path = os.path.join(self.models_dir, f"{model_id}.npy")
            if not os.path.exists(model_path):
                continue
            eligible.append(sym)

        if not eligible:
            return []

        sig = pd.to_datetime(signal_date)
        # Need enough history for MA50/RSI14/vol20. 80 trading days is fine.
        start = (sig - timedelta(days=120)).strftime("%Y-%m-%d")
        sig_str = sig.strftime("%Y-%m-%d")

        # SQLite placeholder limit is fine for 503.
        placeholders = ",".join(["?"] * len(eligible))
        q = (
            "SELECT symbol, date, open, high, low, close, volume "
            "FROM prices "
            f"WHERE symbol IN ({placeholders}) AND date >= ? "
            "ORDER BY symbol, date"
        )
        df = pd.read_sql(q, conn, params=list(eligible) + [start])
        if df.empty:
            return []

        df["date"] = pd.to_datetime(df["date"])

        def _add_feats(g):
            g = g.sort_values("date").reset_index(drop=True)
            g["return_1d"] = g["close"].pct_change(1)
            g["return_5d"] = g["close"].pct_change(5)
            g["return_10d"] = g["close"].pct_change(10)
            g["volatility_20d"] = g["return_1d"].rolling(20).std()
            g["ma_20"] = g["close"].rolling(20).mean()
            g["ma_50"] = g["close"].rolling(50).mean()
            g["dist_ma_20"] = (g["close"] - g["ma_20"]) / g["ma_20"]
            g["dist_ma_50"] = (g["close"] - g["ma_50"]) / g["ma_50"]

            delta = g["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            g["rsi_14"] = 100 - (100 / (1 + rs))

            g["volume_ma_20"] = g["volume"].rolling(20).mean()
            g["volume_ratio"] = g["volume"] / g["volume_ma_20"]

            # News features default to 0 if you don't ingest news for all 503 daily.
            g["news_count_7d"] = 0
            g["news_sentiment_7d"] = 0.0
            return g

        feat = df.groupby("symbol", group_keys=False).apply(_add_feats)
        # Pull features for the signal_date.
        feat_on_date = feat[feat["date"] == pd.to_datetime(sig_str)]
        if feat_on_date.empty:
            return []
        feat_on_date = feat_on_date.set_index("symbol")

        rankings = []
        for sym in eligible:
            row = feat_on_date.loc[[sym]] if sym in feat_on_date.index else None
            if row is None or row.empty:
                continue
            reg_entry = registry.get(sym) or {}
            model_id = reg_entry.get("latest_model")
            features_list = reg_entry.get("features") or []
            if not model_id or not features_list:
                continue
            try:
                coeffs = np.load(os.path.join(self.models_dir, f"{model_id}.npy"))
            except Exception:
                continue
            missing = [c for c in features_list if c not in row.columns]
            if missing:
                continue
            try:
                X = row[features_list].values[0]
                X_bias = np.insert(X, 0, 1.0)
                pred_return = float(X_bias @ coeffs)
            except Exception:
                continue
            rankings.append({"symbol": sym, "predicted_return": pred_return})

        return rankings

    def get_fallback_rankings_for_date_bulk(self, symbols, signal_date, conn, lookback_days=120):
        """
        Fallback ranking when no ML models are available.
        Uses simple 10-day momentum computed from price history.
        """
        sig = pd.to_datetime(signal_date)
        start = (sig - timedelta(days=int(lookback_days))).strftime("%Y-%m-%d")
        sig_str = sig.strftime("%Y-%m-%d")

        symbols = [str(s or "").strip().upper() for s in symbols if str(s or "").strip()]
        if not symbols:
            return []

        placeholders = ",".join(["?"] * len(symbols))
        q = (
            "SELECT symbol, date, close "
            "FROM prices "
            f"WHERE symbol IN ({placeholders}) AND date >= ? "
            "ORDER BY symbol, date"
        )
        df = pd.read_sql(q, conn, params=list(symbols) + [start])
        if df.empty:
            return []

        df["date"] = pd.to_datetime(df["date"])

        def _add_momentum(g):
            g = g.sort_values("date").reset_index(drop=True)
            g["return_10d"] = g["close"].pct_change(10)
            return g

        feat = df.groupby("symbol", group_keys=False).apply(_add_momentum)
        feat_on_date = feat[feat["date"] == pd.to_datetime(sig_str)]
        if feat_on_date.empty:
            return []

        rankings = []
        for _, row in feat_on_date.iterrows():
            sym = str(row.get("symbol", "")).strip().upper()
            if not sym:
                continue
            try:
                score = float(row.get("return_10d", 0.0))
            except Exception:
                continue
            rankings.append({"symbol": sym, "predicted_return": score})
        return rankings

    def run_daily_test(self, test_date=None, pipeline_stats=None, backtest_signals=None):
        """
        Position-Tracking Backtest:
        1. Open new positions based on signals.
        2. Check existing positions for TP hits.
        3. Report unrealized P&L.
        """
        conn = sqlite3.connect(self.db_path)

        if test_date is None:
            last_date = pd.read_sql("SELECT MAX(date) as max_date FROM prices", conn).iloc[0, 0]
            if not last_date:
                logger.warning("No price data available for backtest.")
                conn.close()
                return None
            test_date = pd.to_datetime(last_date)
        else:
            test_date = pd.to_datetime(test_date)

        # signal_date is the day BEFORE test_date
        signal_date = test_date - timedelta(days=1)

        # Handle weekends
        attempts = 0
        while attempts < 5:
            check = pd.read_sql(
                f"SELECT COUNT(*) as cnt FROM prices WHERE date='{signal_date.strftime('%Y-%m-%d')}'",
                conn
            )
            if check.iloc[0]['cnt'] > 0:
                break
            signal_date = signal_date - timedelta(days=1)
            attempts += 1

        logger.info(f"Running position-tracking backtest for {test_date.date()}")
        logger.info(f"Using signals from {signal_date.date()} (T-1)")

        # Load universe
        universe_df = pd.read_csv(self.universe_path)

        # Generate rankings using T-1 features
        rankings = self.get_predictions_for_date_bulk(
            universe_df["ticker"].tolist(),
            signal_date,
            conn,
        )

        if not rankings:
            logger.warning("No ML predictions generated; falling back to momentum-based rankings.")
            rankings = self.get_fallback_rankings_for_date_bulk(
                universe_df["ticker"].tolist(),
                signal_date,
                conn,
            )
            if not rankings:
                logger.warning("No fallback rankings generated.")
                conn.close()
                return None

        rank_df = pd.DataFrame(rankings).sort_values('predicted_return', ascending=False)

        # Apply meta-learner adjustments
        from meta_learner import MetaLearner
        meta = MetaLearner(self.config_path)
        # IMPORTANT: refresh meta-learner state from recent closed trades.
        # Without this, meta_learner_state.json can become stale and repeat the same tickers/penalties forever.
        try:
            ml_cfg = self.config.get("meta_learning", {}) if isinstance(self.config, dict) else {}
            lookback_days = int(ml_cfg.get("lookback_days", 30))
            meta.analyze_past_trades(lookback_days=lookback_days)
        except Exception as exc:
            logger.warning(f"Meta-learner analysis failed; continuing without updated penalties: {exc}")
        rank_df = meta.get_confidence_adjustments(rank_df)
        rank_df = rank_df.sort_values('adjusted_score', ascending=False).reset_index(drop=True)
        rank_df['rank'] = rank_df.index + 1
        rank_lookup = rank_df.set_index('symbol')[['predicted_return', 'adjusted_score', 'penalty', 'rank']]

        # Capture Meta-Learner Insights
        meta_insights = meta.get_daily_insights()

        # Determine Current Capital (Compounding)
        initial_capital = 100000
        summary = self.core_tracker.get_portfolio_summary()
        total_realized_dollars = summary.get('total_realized_pnl_dollars', 0.0)
        total_unrealized_dollars = summary.get('total_unrealized_pnl_dollars', 0.0)
        current_capital = initial_capital + total_realized_dollars
        capital_start = current_capital

        # Portfolio selection (top K)
        open_positions_df = self.core_tracker.get_open_positions()
        open_symbols = set(open_positions_df['symbol']) if not open_positions_df.empty else set()
        max_positions = self.config.get('trading', {}).get('max_positions', 10)
        trading_cfg = self.config.get("trading", {}) if isinstance(self.config, dict) else {}
        risk_cfg = self.config.get("risk", {}) if isinstance(self.config, dict) else {}
        short_threshold = float(trading_cfg.get("short_threshold", -0.0))
        available_slots = max(0, max_positions - len(open_symbols))
        invested_cost = 0.0
        if not open_positions_df.empty:
            invested_cost = (open_positions_df['entry_price'] * open_positions_df['quantity']).sum()

        available_capital = max(0.0, current_capital - invested_cost)
        cash_pct = (available_capital / current_capital) if current_capital else 0.0
        max_cash_pct = float(trading_cfg.get("max_cash_pct", 1.0))
        max_cash_pct = max(0.0, min(1.0, max_cash_pct))
        min_weight = float(trading_cfg.get("min_total_weight", 0.0) or 0.0)
        min_weight = max(0.0, min(1.0, min_weight))
        idle_cash_caps = [float(current_capital) * max_cash_pct]
        if min_weight > 0.0:
            idle_cash_caps.append(float(current_capital) * max(0.0, 1.0 - min_weight))
        allowed_idle_cash = min(idle_cash_caps) if idle_cash_caps else 0.0
        cash_drag_excess = available_capital > 0 and cash_pct > max_cash_pct

        if available_slots == 0 or available_capital <= 0:
            if available_slots == 0 and cash_drag_excess:
                logger.warning(
                    "Cash drag %.1f%% exceeds max_cash_pct %.1f%% but no slots are available (max_positions reached).",
                    cash_pct * 100,
                    max_cash_pct * 100,
                )
            logger.info("No available slots or capital for new positions today.")
            target_portfolio = pd.DataFrame()
        else:
            rank_df_filtered = rank_df[~rank_df['symbol'].isin(open_symbols)]
            # Also avoid recently exited names (cooldown) to prevent repeating the same tickers.
            blocked = meta.get_exit_cooldown_symbols(as_of_date=signal_date)
            if blocked:
                rank_df_filtered = rank_df_filtered[~rank_df_filtered['symbol'].isin(blocked)]
            portfolio_mgr = PortfolioManager(self.config_path)
            enable_shorts = bool(self.config.get("trading", {}).get("enable_shorts", False))
            max_shorts = int(self.config.get("trading", {}).get("max_shorts", 0))

            long_df = rank_df_filtered[rank_df_filtered["adjusted_score"] > 0].copy()
            short_df = pd.DataFrame()
            if enable_shorts and max_shorts > 0:
                short_df = rank_df_filtered[rank_df_filtered["adjusted_score"] <= short_threshold].copy()

            # Cash-drag fallback: if no signals and cash is too high, relax filters.
            if cash_drag_excess and long_df.empty and short_df.empty:
                logger.warning(
                    "Cash drag %.1f%% exceeds max_cash_pct %.1f%%; relaxing signal filters to deploy capital.",
                    cash_pct * 100,
                    max_cash_pct * 100,
                )
                long_df = rank_df_filtered.copy()
                if enable_shorts and max_shorts > 0:
                    short_df = rank_df_filtered[rank_df_filtered["adjusted_score"] < 0].copy()
                    if not short_df.empty:
                        long_df = long_df[~long_df["symbol"].isin(short_df["symbol"])]

            # Allocate slots between longs and shorts (shorts capped).
            shorts_slots = min(max_shorts, available_slots)
            long_slots = max(0, available_slots - shorts_slots)
            target_portfolio = portfolio_mgr.generate_target_portfolio(long_df, max_positions=long_slots)

            if not short_df.empty and shorts_slots > 0:
                # Take the most negative adjusted scores (strongest short signals)
                short_df = short_df.sort_values("adjusted_score", ascending=True).head(shorts_slots).copy()
                # Size shorts using absolute adjusted score
                if portfolio_mgr.equal_weight:
                    short_df["weight"] = 1.0 / len(short_df)
                else:
                    scores = (-short_df["adjusted_score"]).clip(lower=0)
                    total = scores.sum()
                    short_df["weight"] = (scores / total) if total else (1.0 / len(short_df))
                short_df["side"] = "SHORT"

                if target_portfolio is None or target_portfolio.empty:
                    target_portfolio = short_df
                else:
                    target_portfolio = pd.concat([target_portfolio, short_df], ignore_index=True)

            # Guardrail: combined long+short weights must never exceed 100% total capital.
            if target_portfolio is not None and not target_portfolio.empty:
                target_portfolio["weight"] = pd.to_numeric(
                    target_portfolio["weight"], errors="coerce"
                ).fillna(0.0).clip(lower=0.0)
                total_weight = float(target_portfolio["weight"].sum() or 0.0)
                if total_weight > 1.0:
                    target_portfolio["weight"] = target_portfolio["weight"] / total_weight

                # Guardrail: enforce minimum total weight to avoid leaving too much cash idle
                if min_weight > 0 and 0 < total_weight < min_weight:
                    target_portfolio["weight"] = target_portfolio["weight"] * (min_weight / total_weight)
                    logger.info(f"Adjusted weights from {total_weight:.2%} to {min_weight:.2%} to deploy more capital")

        # Open NEW positions for selected stocks
        tp_pct = self.config.get('trading', {}).get('take_profit_pct', 0.03)
        new_positions = []
        top_up_positions = []
        remaining_capital = float(available_capital)
        for _, row in target_portfolio.iterrows():
            symbol = row['symbol']
            raw_side = row.get("side", None)
            if raw_side is None or (isinstance(raw_side, float) and pd.isna(raw_side)):
                side = "LONG"
            else:
                side = str(raw_side).strip().upper() or "LONG"
            if side not in {"LONG", "SHORT"}:
                side = "LONG"

            # Get entry price (Day T Open)
            price_data = pd.read_sql(
                f"SELECT * FROM prices WHERE symbol='{symbol}' AND date='{test_date.strftime('%Y-%m-%d')}'",
                conn
            )

            if not price_data.empty:
                entry_price = price_data.iloc[0]['open']
                requested_allocation = max(
                    0.0, float(row.get('weight', 0.0) or 0.0) * float(available_capital)
                )
                allocation_dollars = min(requested_allocation, max(0.0, remaining_capital))
                if allocation_dollars <= 0.0:
                    continue
                quantity = allocation_dollars / entry_price
                allocation_pct = (allocation_dollars / current_capital * 100) if current_capital else 0.0

                pos_id = self.core_tracker.open_position(
                    symbol=symbol,
                    entry_date=test_date.strftime('%Y-%m-%d'),
                    entry_price=entry_price,
                    quantity=quantity,
                    side=side
                )

                if pos_id:
                    reason = "Top-ranked model signal"
                    if symbol in rank_lookup.index:
                        info = rank_lookup.loc[symbol]
                        pred = float(info['predicted_return'])
                        adj = float(info['adjusted_score'])
                        penalty = float(info['penalty'])
                        rank = int(info['rank'])
                        reason = (
                            f"Rank {rank} signal (pred {pred:.2%}, adj {adj:.2%}, penalty {penalty:.2f})"
                        )

                    new_positions.append({
                        'symbol': symbol,
                        'side': side,
                        'entry_price': entry_price,
                        'target_price': entry_price * (1 + tp_pct) if side == "LONG" else entry_price * (1 - tp_pct),
                        'quantity': quantity,
                        'allocation_pct': allocation_pct,
                        'allocation_dollars': allocation_dollars,
                        'reason': reason
                    })
                    remaining_capital = max(0.0, remaining_capital - allocation_dollars)

        # If older under-sized positions are occupying slots, scale into the best-held names
        # before leaving excess cash idle.
        extra_to_deploy = max(0.0, float(remaining_capital) - float(allowed_idle_cash))
        max_position_equity_pct = float(risk_cfg.get("max_position_equity_pct", 1.0) or 1.0)
        max_position_equity_pct = max(0.0, min(1.0, max_position_equity_pct))
        if extra_to_deploy > 0.0 and max_position_equity_pct > 0.0:
            open_positions_for_top_up = self.core_tracker.get_open_positions()
            if open_positions_for_top_up is not None and not open_positions_for_top_up.empty:
                top_up_df = open_positions_for_top_up.copy()
                top_up_df["symbol"] = top_up_df["symbol"].astype(str).str.strip().str.upper()
                top_up_df["side"] = top_up_df["side"].fillna("LONG").astype(str).str.upper()
                top_up_df["entry_price"] = pd.to_numeric(top_up_df["entry_price"], errors="coerce").fillna(0.0)
                top_up_df["quantity"] = pd.to_numeric(top_up_df["quantity"], errors="coerce").fillna(0.0)
                top_up_df["notional"] = top_up_df["entry_price"] * top_up_df["quantity"]
                top_up_df["adjusted_score"] = top_up_df["symbol"].map(
                    lambda sym: float(rank_lookup.loc[sym]["adjusted_score"]) if sym in rank_lookup.index else 0.0
                )
                top_up_df["rank_priority"] = top_up_df["adjusted_score"].abs()

                supported_mask = (
                    ((top_up_df["side"] == "LONG") & (top_up_df["adjusted_score"] > 0))
                    | ((top_up_df["side"] == "SHORT") & (top_up_df["adjusted_score"] <= short_threshold))
                )
                top_up_candidates = top_up_df[supported_mask].copy()
                if top_up_candidates.empty:
                    top_up_candidates = top_up_df.copy()

                top_up_candidates = top_up_candidates.sort_values(
                    ["rank_priority", "adjusted_score"],
                    ascending=[False, False],
                )

                max_position_notional = float(current_capital) * max_position_equity_pct
                logger.warning(
                    "Core cash drag remains %.2f with idle-cash cap %.2f; topping up existing positions.",
                    float(remaining_capital),
                    float(allowed_idle_cash),
                )
                for _, pos in top_up_candidates.iterrows():
                    if extra_to_deploy <= 0.0:
                        break

                    current_notional = float(pos.get("notional", 0.0) or 0.0)
                    room = max(0.0, max_position_notional - current_notional)
                    if room <= 0.0:
                        continue

                    symbol = str(pos["symbol"]).strip().upper()
                    side = str(pos.get("side") or "LONG").upper()
                    price_data = pd.read_sql(
                        "SELECT open FROM prices WHERE symbol=? AND date=?",
                        conn,
                        params=(symbol, test_date.strftime("%Y-%m-%d")),
                    )
                    if price_data.empty:
                        continue

                    entry_price = float(price_data.iloc[0]["open"] or 0.0)
                    if entry_price <= 0.0:
                        continue

                    allocation_dollars = min(extra_to_deploy, room)
                    quantity = allocation_dollars / entry_price if entry_price else 0.0
                    if quantity <= 0.0:
                        continue

                    added = self.core_tracker.add_to_position(
                        symbol=symbol,
                        add_date=test_date.strftime("%Y-%m-%d"),
                        add_price=entry_price,
                        quantity=quantity,
                        side=side,
                    )
                    if not added:
                        continue

                    reason = "Cash-drag top-up of existing position"
                    if symbol in rank_lookup.index:
                        info = rank_lookup.loc[symbol]
                        reason = (
                            f"Cash-drag top-up (pred {float(info['predicted_return']):.2%}, "
                            f"adj {float(info['adjusted_score']):.2%}, rank {int(info['rank'])})"
                        )

                    top_up_positions.append({
                        "symbol": symbol,
                        "side": side,
                        "entry_price": entry_price,
                        "target_price": added["target_price"],
                        "quantity": quantity,
                        "allocation_pct": (allocation_dollars / current_capital * 100) if current_capital else 0.0,
                        "allocation_dollars": allocation_dollars,
                        "reason": reason,
                    })
                    remaining_capital = max(0.0, remaining_capital - allocation_dollars)
                    extra_to_deploy = max(0.0, extra_to_deploy - allocation_dollars)

                if extra_to_deploy > 0.0:
                    logger.warning(
                        "Core cash drag persists after top-ups; %.2f cash still exceeds the idle-cash cap.",
                        extra_to_deploy,
                    )

        entries_today = list(new_positions) + list(top_up_positions)

        # Keep Core + AI strategies distinct by optionally preventing overlap.
        core_reserved_symbols = set(open_symbols)
        if target_portfolio is not None and hasattr(target_portfolio, "empty") and not target_portfolio.empty:
            try:
                core_reserved_symbols |= set([str(s).strip().upper() for s in target_portfolio["symbol"].tolist() if str(s).strip()])
            except Exception:
                pass

        conn.close()

        # Check ALL open positions for TP hits on this day
        closed_positions = self.core_tracker.check_and_close_positions(
            check_date=test_date.strftime('%Y-%m-%d')
        )

        # Save closed trades for meta-learning (per-day)
        os.makedirs(self.results_dir, exist_ok=True)
        date_str = test_date.strftime('%Y%m%d')
        if closed_positions:
            trades_df = pd.DataFrame(closed_positions)
            trades_df = trades_df.rename(columns={
                "realized_pnl": "strat_return",
                "realized_pnl_dollars": "pnl_dollars",
            })
            trades_df.to_csv(
                os.path.join(self.results_dir, f"trades_{date_str}.csv"),
                index=False
            )

        # Get updated portfolio state
        summary = self.core_tracker.get_portfolio_summary()
        unrealized = self.core_tracker.get_unrealized_pnl()
        total_realized_dollars = summary.get('total_realized_pnl_dollars', 0.0)
        total_unrealized_dollars = summary.get('total_unrealized_pnl_dollars', 0.0)

        # Save daily report
        os.makedirs(self.results_dir, exist_ok=True)
        date_str = test_date.strftime('%Y%m%d')

        # Calculate Realized P&L for TODAY
        realized_today_dollars = sum([p.get('realized_pnl_dollars', 0.0) for p in closed_positions])
        realized_today = (realized_today_dollars / capital_start) if capital_start else 0.0

        total_realized_pct = (total_realized_dollars / initial_capital) if initial_capital else 0.0
        total_unrealized_pct = (total_unrealized_dollars / initial_capital) if initial_capital else 0.0
        total_account_return = ((total_realized_dollars + total_unrealized_dollars) / initial_capital) if initial_capital else 0.0
        current_capital = initial_capital + total_realized_dollars

        # Available cash (notional-based; shorts consume capital too)
        open_positions_now = self.core_tracker.get_open_positions()
        invested_notional = 0.0
        if not open_positions_now.empty:
            invested_notional = float((open_positions_now["entry_price"] * open_positions_now["quantity"]).sum() or 0.0)
        available_cash = float(current_capital) - invested_notional

        report = {
            'date': test_date.date(),
            'new_positions_opened': len(new_positions),
            'positions_topped_up': len(top_up_positions),
            'positions_closed_at_tp': len(closed_positions),
            'open_positions': summary['open_positions'],
            'realized_pnl_today': realized_today,
            'realized_pnl_today_dollars': realized_today_dollars,
            'total_realized_pnl': total_realized_pct,
            'total_realized_pnl_dollars': total_realized_dollars,
            'total_unrealized_pnl': total_unrealized_pct,
            'total_unrealized_pnl_dollars': total_unrealized_dollars,
            'total_account_return': total_account_return,
            'current_capital_estimate': current_capital,
            'invested_notional': invested_notional,
            'available_cash': available_cash,
            'initial_capital': initial_capital
        }

        pd.DataFrame([report]).to_csv(
            os.path.join(self.results_dir, f"daily_report_{date_str}.csv"),
            index=False
        )

        if not unrealized.empty:
            unrealized.to_csv(
                os.path.join(self.results_dir, f"unrealized_{date_str}.csv"),
                index=False
            )

        logger.info(f"Daily report saved to {self.results_dir}")

        # --- AI strategy (separate $100k account) ---
        from llm_trader import propose_trades_with_llm

        ai_cfg = self.config.get("ai_trading", {})
        ai_enabled = bool(ai_cfg.get("enabled", False))
        ai_report = None
        ai_unrealized = pd.DataFrame()
        ai_closed = []
        ai_new = []
        ai_llm_status = {"enabled": False, "ok": False, "error": "disabled"}

        if ai_enabled:
            ai_initial_capital = float(ai_cfg.get("initial_capital", 100000))
            # Capture pre-close capital for reporting, then close TP positions before
            # deciding new AI entries so freed slots/cash are usable in the same run.
            ai_summary_pre_close = self.ai_tracker.get_portfolio_summary()
            ai_realized_pre_close = float(ai_summary_pre_close.get("total_realized_pnl_dollars", 0.0))
            ai_capital_pre_close = ai_initial_capital + ai_realized_pre_close

            ai_closed = self.ai_tracker.check_and_close_positions(check_date=test_date.strftime('%Y-%m-%d'))

            ai_summary0 = self.ai_tracker.get_portfolio_summary()
            ai_realized_total_dollars0 = float(ai_summary0.get("total_realized_pnl_dollars", 0.0))
            ai_current_capital0 = ai_initial_capital + ai_realized_total_dollars0
            ai_open_df = self.ai_tracker.get_open_positions()
            ai_open_symbols = set(ai_open_df["symbol"]) if not ai_open_df.empty else set()
            ai_invested_cost = (ai_open_df["entry_price"] * ai_open_df["quantity"]).sum() if not ai_open_df.empty else 0.0
            ai_available_capital = max(0.0, ai_current_capital0 - ai_invested_cost)

            ai_max_positions = int(ai_cfg.get("max_positions", 10))
            ai_allow_shorts = bool(ai_cfg.get("allow_shorts", True))
            ai_max_shorts = int(ai_cfg.get("max_shorts", 5))
            ai_available_slots = max(0, ai_max_positions - len(ai_open_symbols))

            # AI strategy uses the trained model only. Core rankings remain untouched.
            def _pyfloat(value):
                try:
                    if pd.isna(value):
                        return None
                    return float(value)
                except Exception:
                    return None

            def _recent_price_metrics(conn_, sym_, lookback=30):
                dfp = pd.read_sql(
                    "SELECT date, close, volume FROM prices WHERE symbol=? ORDER BY date DESC LIMIT ?",
                    conn_,
                    params=(sym_, int(lookback)),
                )
                if dfp.empty or len(dfp) < 2:
                    return None
                dfp = dfp.sort_values("date").reset_index(drop=True)
                closes = dfp["close"].astype(float).tolist()
                tail_n = min(10, len(closes))
                closes_tail = [float(x) for x in closes[-tail_n:]]
                v20 = float(dfp["volume"].astype(float).tail(20).mean()) if "volume" in dfp.columns else None
                v1 = float(dfp["volume"].astype(float).iloc[-1]) if "volume" in dfp.columns else None
                return {
                    "last_date": str(dfp["date"].iloc[-1]),
                    "last_close": float(closes_tail[-1]),
                    "closes_tail": closes_tail,
                    "volume_1d": v1,
                    "volume_20d_avg": v20,
                }

            def _latest_feature_snapshot(conn_, sym_, lookback=80, news_window_days=7):
                dfp = pd.read_sql(
                    "SELECT date, close, volume FROM prices WHERE symbol=? ORDER BY date DESC LIMIT ?",
                    conn_,
                    params=(sym_, int(lookback)),
                )
                if dfp.empty or len(dfp) < 20:
                    return {
                        "return_1d": None,
                        "return_5d": None,
                        "return_10d": None,
                        "volatility_20d": None,
                        "dist_ma_20": None,
                        "dist_ma_50": None,
                        "rsi_14": None,
                        "volume_ratio": None,
                        "news_count_7d": 0,
                        "news_sentiment_7d": 0.0,
                    }

                dfp = dfp.sort_values("date").reset_index(drop=True).copy()
                dfp["close"] = pd.to_numeric(dfp["close"], errors="coerce")
                dfp["volume"] = pd.to_numeric(dfp["volume"], errors="coerce")
                dfp["return_1d"] = dfp["close"].pct_change(1)
                dfp["return_5d"] = dfp["close"].pct_change(5)
                dfp["return_10d"] = dfp["close"].pct_change(10)
                dfp["volatility_20d"] = dfp["return_1d"].rolling(20).std()
                dfp["ma_20"] = dfp["close"].rolling(20).mean()
                dfp["ma_50"] = dfp["close"].rolling(50).mean()
                dfp["dist_ma_20"] = (dfp["close"] - dfp["ma_20"]) / dfp["ma_20"]
                dfp["dist_ma_50"] = (dfp["close"] - dfp["ma_50"]) / dfp["ma_50"]

                delta = dfp["close"].diff()
                gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
                rs = gain / loss.replace(0, pd.NA)
                dfp["rsi_14"] = 100 - (100 / (1 + rs))
                dfp["volume_ma_20"] = dfp["volume"].rolling(20).mean()
                dfp["volume_ratio"] = dfp["volume"] / dfp["volume_ma_20"]

                latest = dfp.iloc[-1]
                end_dt = pd.to_datetime(latest["date"])
                news_count_7d = 0
                news_sentiment_7d = 0.0
                try:
                    news_df = pd.read_sql(
                        "SELECT datetime, sentiment_score FROM news WHERE symbol=? ORDER BY datetime DESC LIMIT 250",
                        conn_,
                        params=(sym_,),
                    )
                    if not news_df.empty and "datetime" in news_df.columns:
                        news_df["datetime"] = pd.to_datetime(news_df["datetime"], errors="coerce")
                        news_df = news_df.dropna(subset=["datetime"])
                        if not news_df.empty:
                            start_dt = end_dt - pd.Timedelta(days=int(news_window_days or 7))
                            news_df = news_df[news_df["datetime"] >= start_dt]
                            news_count_7d = int(len(news_df))
                            if "sentiment_score" in news_df.columns:
                                sentiment = pd.to_numeric(news_df["sentiment_score"], errors="coerce").dropna()
                                if not sentiment.empty:
                                    news_sentiment_7d = float(sentiment.mean())
                except Exception:
                    news_count_7d = 0
                    news_sentiment_7d = 0.0

                return {
                    "return_1d": _pyfloat(latest.get("return_1d")),
                    "return_5d": _pyfloat(latest.get("return_5d")),
                    "return_10d": _pyfloat(latest.get("return_10d")),
                    "volatility_20d": _pyfloat(latest.get("volatility_20d")),
                    "dist_ma_20": _pyfloat(latest.get("dist_ma_20")),
                    "dist_ma_50": _pyfloat(latest.get("dist_ma_50")),
                    "rsi_14": _pyfloat(latest.get("rsi_14")),
                    "volume_ratio": _pyfloat(latest.get("volume_ratio")),
                    "news_count_7d": int(news_count_7d),
                    "news_sentiment_7d": float(news_sentiment_7d),
                }

            universe_symbols = [str(t).strip().upper() for t in universe_df["ticker"].tolist() if str(t).strip()]
            universe_symbols = [s for s in universe_symbols if s not in ai_open_symbols]

            disallow_overlap = bool(ai_cfg.get("disallow_core_overlap", True))
            blocked_by_core = 0
            if disallow_overlap and core_reserved_symbols:
                before = len(universe_symbols)
                universe_symbols = [s for s in universe_symbols if s not in core_reserved_symbols]
                blocked_by_core = before - len(universe_symbols)

            conn_ai = sqlite3.connect(self.db_path)
            cand = []
            try:
                seed = f"{pd.to_datetime(signal_date).date().isoformat()}-ai"
                rng = random.Random(seed)
                rng.shuffle(universe_symbols)

                prompt_limit = int(ai_cfg.get("prompt_candidates_limit", 200) or 200)
                price_lookback = int(ai_cfg.get("price_lookback_days", 30) or 30)
                feature_lookback = int(ai_cfg.get("feature_lookback_days", 80) or 80)
                news_window_days = int(ai_cfg.get("news_window_days", 7) or 7)

                for sym in universe_symbols:
                    if len(cand) >= max(1, prompt_limit):
                        break
                    snapshot = _recent_price_metrics(conn_ai, sym, lookback=price_lookback)
                    if not snapshot:
                        continue
                    snapshot.update(
                        _latest_feature_snapshot(
                            conn_ai,
                            sym,
                            lookback=feature_lookback,
                            news_window_days=news_window_days,
                        )
                    )
                    snapshot["symbol"] = sym
                    snapshot["as_of_date"] = str(pd.to_datetime(signal_date).date())
                    cand.append(snapshot)
            finally:
                conn_ai.close()

            if ai_available_capital <= 0.0 or ai_available_slots <= 0:
                ai_trades = []
                ai_llm_status = {
                    "enabled": True,
                    "ok": True,
                    "skipped_reason": "no_capacity",
                    "error": None,
                    "model": ((ai_cfg.get("trained_model") or {}).get("base_model") or (ai_cfg.get("trained_model") or {}).get("inference_url")),
                    "model_used": None,
                    "disallow_core_overlap": disallow_overlap,
                    "blocked_by_core": blocked_by_core,
                    "candidates_built": len(cand),
                    "available_capital": float(ai_available_capital),
                    "available_slots": int(ai_available_slots),
                }
            else:
                ai_trades, ai_llm_status = propose_trades_with_llm(
                    self.config,
                    cand,
                    max_positions=ai_available_slots,
                    allow_shorts=ai_allow_shorts,
                    max_shorts=ai_max_shorts,
                )
                if isinstance(ai_llm_status, dict):
                    ai_llm_status["disallow_core_overlap"] = disallow_overlap
                    ai_llm_status["blocked_by_core"] = blocked_by_core
                    ai_llm_status["candidates_built"] = len(cand)
                # Strict gate: no LLM success -> no new AI entries.
                if not (isinstance(ai_llm_status, dict) and ai_llm_status.get("ok")):
                    ai_trades = []
                    if isinstance(ai_llm_status, dict):
                        ai_llm_status["entries_blocked_due_to_llm_error"] = True

            if ai_trades and ai_available_capital > 0:
                conn_ai = sqlite3.connect(self.db_path)
                ai_remaining_capital = float(ai_available_capital)
                for t in ai_trades:
                    symbol = t["symbol"]
                    side = str(t.get("side") or "LONG").upper()
                    weight = float(t.get("weight", 0.0) or 0.0)
                    requested_allocation = max(0.0, weight * float(ai_available_capital))
                    allocation_dollars = min(requested_allocation, max(0.0, ai_remaining_capital))
                    if allocation_dollars <= 0:
                        continue

                    price_data = pd.read_sql(
                        f"SELECT * FROM prices WHERE symbol='{symbol}' AND date='{test_date.strftime('%Y-%m-%d')}'",
                        conn_ai
                    )
                    if price_data.empty:
                        continue
                    entry_price = float(price_data.iloc[0]["open"])
                    quantity = allocation_dollars / entry_price if entry_price else 0.0
                    if quantity <= 0:
                        continue

                    pos_id = self.ai_tracker.open_position(
                        symbol=symbol,
                        entry_date=test_date.strftime('%Y-%m-%d'),
                        entry_price=entry_price,
                        quantity=quantity,
                        side=side
                    )
                    if pos_id:
                        ai_new.append({
                            "symbol": symbol,
                            "side": side,
                            "entry_price": entry_price,
                            "target_price": entry_price * (1 + tp_pct) if side == "LONG" else entry_price * (1 - tp_pct),
                            "quantity": quantity,
                            "allocation_pct": (allocation_dollars / ai_current_capital0 * 100) if ai_current_capital0 else 0.0,
                            "allocation_dollars": allocation_dollars,
                            "reason": t.get("reason") or "LLM trade",
                        })
                        ai_remaining_capital = max(0.0, ai_remaining_capital - allocation_dollars)
                conn_ai.close()

            ai_summary = self.ai_tracker.get_portfolio_summary()
            ai_unrealized = self.ai_tracker.get_unrealized_pnl()
            ai_realized_total_dollars = float(ai_summary.get("total_realized_pnl_dollars", 0.0))
            ai_unreal_total_dollars = float(ai_summary.get("total_unrealized_pnl_dollars", 0.0))
            ai_realized_today_dollars = sum([p.get('realized_pnl_dollars', 0.0) for p in ai_closed])
            ai_realized_today = (ai_realized_today_dollars / ai_capital_pre_close) if ai_capital_pre_close else 0.0

            ai_current_capital = ai_initial_capital + ai_realized_total_dollars
            ai_open_now = self.ai_tracker.get_open_positions()
            ai_invested_notional = 0.0
            if not ai_open_now.empty:
                ai_invested_notional = float((ai_open_now["entry_price"] * ai_open_now["quantity"]).sum() or 0.0)
            ai_available_cash = float(ai_current_capital) - ai_invested_notional

            ai_report = {
                "date": test_date.date(),
                "new_positions_opened": len(ai_new),
                "positions_closed_at_tp": len(ai_closed),
                "open_positions": int(ai_summary.get("open_positions", 0)),
                "realized_pnl_today": ai_realized_today,
                "realized_pnl_today_dollars": ai_realized_today_dollars,
                "total_realized_pnl": (ai_realized_total_dollars / ai_initial_capital) if ai_initial_capital else 0.0,
                "total_realized_pnl_dollars": ai_realized_total_dollars,
                "total_unrealized_pnl": (ai_unreal_total_dollars / ai_initial_capital) if ai_initial_capital else 0.0,
                "total_unrealized_pnl_dollars": ai_unreal_total_dollars,
                "total_account_return": ((ai_realized_total_dollars + ai_unreal_total_dollars) / ai_initial_capital) if ai_initial_capital else 0.0,
                "current_capital_estimate": ai_current_capital,
                "invested_notional": ai_invested_notional,
                "available_cash": ai_available_cash,
                "initial_capital": ai_initial_capital,
                "ai_llm_ok": bool((ai_llm_status or {}).get("ok")) if isinstance(ai_llm_status, dict) else False,
                "ai_llm_error": str((ai_llm_status or {}).get("error") or "") if isinstance(ai_llm_status, dict) else "",
                "ai_llm_skipped_reason": str((ai_llm_status or {}).get("skipped_reason") or "") if isinstance(ai_llm_status, dict) else "",
                "ai_model_used": str((ai_llm_status or {}).get("model_used") or "") if isinstance(ai_llm_status, dict) else "",
                "ai_candidates_seen": int((ai_llm_status or {}).get("candidates_seen") or (ai_llm_status or {}).get("candidates_built") or 0) if isinstance(ai_llm_status, dict) else 0,
                "ai_candidates_scored": int((ai_llm_status or {}).get("candidates_scored") or 0) if isinstance(ai_llm_status, dict) else 0,
            }

            pd.DataFrame([ai_report]).to_csv(
                os.path.join(self.results_dir, f"daily_report_ai_{date_str}.csv"),
                index=False
            )
            if not ai_unrealized.empty:
                ai_unrealized.to_csv(
                    os.path.join(self.results_dir, f"unrealized_ai_{date_str}.csv"),
                    index=False
                )

        if pipeline_stats is None:
            pipeline_stats = {}
        if isinstance(pipeline_stats, dict):
            if ai_enabled and isinstance(ai_llm_status, dict) and not bool(ai_llm_status.get("ok")):
                _record_pipeline_issue(
                    pipeline_stats,
                    "ERROR",
                    "AI Trading",
                    ai_llm_status.get("error") or "AI trading decision call failed.",
                )
            _finalize_pipeline_health(pipeline_stats)

        # Keep Core and AI reporting stats separate to avoid showing AI-trading fields
        # in the Core email.
        core_pipeline_stats = dict(pipeline_stats) if isinstance(pipeline_stats, dict) else pipeline_stats
        ai_pipeline_stats = dict(pipeline_stats) if isinstance(pipeline_stats, dict) else pipeline_stats
        if isinstance(core_pipeline_stats, dict):
            core_pipeline_stats.pop("ai_trading_llm_status", None)
        if isinstance(ai_pipeline_stats, dict):
            ai_pipeline_stats["ai_trading_llm_status"] = ai_llm_status

        # Send TWO separate emails (Core + AI)
        from email_notifier import EmailNotifier
        notifier = EmailNotifier()
        core_email_sent = notifier.send_daily_report(
            report_data=report,
            unrealized_df=unrealized,
            closed_positions=closed_positions,
            new_positions=entries_today,
            meta_insights=meta_insights,
            signal_rankings=rank_df,
            pipeline_stats=core_pipeline_stats,
            backtest_signals=backtest_signals,
            subject_tag="Core"
        )

        ai_email_sent = True
        if ai_report is not None:
            ai_insight = "AI trading engine status: OK" if ai_llm_status.get("ok") else f"AI trading engine status: ERROR - {ai_llm_status.get('error')}"
            ai_email_sent = notifier.send_daily_report(
                report_data=ai_report,
                unrealized_df=ai_unrealized,
                closed_positions=ai_closed,
                new_positions=ai_new,
                meta_insights=ai_insight,
                signal_rankings=None,
                pipeline_stats=ai_pipeline_stats,
                backtest_signals=backtest_signals,
                subject_tag="AI"
            )

        # A run is considered complete once at least one strategy email is sent.
        # This prevents transient AI-email failures from failing the whole workflow.
        email_sent = bool(core_email_sent) or bool(ai_email_sent)
        if email_sent:
            for marker in [f"email_sent_core_{date_str}.ok", f"email_sent_ai_{date_str}.ok", f"email_sent_{date_str}.ok"]:
                marker_path = os.path.join(self.results_dir, marker)
                try:
                    with open(marker_path, "w") as handle:
                        handle.write(get_sgt_now().isoformat())
                except Exception as exc:
                    logger.warning(f"Failed to write email sent marker: {exc}")
        else:
            logger.warning("One or more emails not sent; scheduler will allow retry.")

        # Cleanup: Keep only last 2 days of files
        self._cleanup_old_files(self.results_dir, keep_days=2)

        return report, unrealized, closed_positions, email_sent

    def _cleanup_old_files(self, results_dir, keep_days=2):
        """Delete result files older than keep_days"""
        now = datetime.now()
        for file in os.listdir(results_dir):
            if not (file.endswith('.csv') or file.endswith('.log')):
                continue
            path = os.path.join(results_dir, file)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if (now - mtime).days > keep_days:
                try:
                    os.remove(path)
                    logger.info(f"Deleted old file: {file}")
                except Exception as exc:
                    logger.warning(f"Failed to delete {file}: {exc}")


def run_full_pipeline(limit=None, config_path=None):
    config_path = _get_config_path(config_path)
    config = _load_config(config_path)
    base_dir = os.path.dirname(os.path.abspath(config_path))
    universe_path = _resolve_path(base_dir, config['universe']['source'])

    universe_df = pd.read_csv(universe_path)
    tickers = universe_df['ticker'].tolist()
    if limit:
        tickers = tickers[:limit]

    logger.info(f"Starting full pipeline for {len(tickers)} stocks...")

    stats = {
        'tickers_total': len(tickers),
        'tickers_processed': 0,
        'tickers_failed': 0,
        'models_trained': 0,
        'news_enabled': config.get('news', {}).get('enabled', False),
        'llm_status': None,
        'error_count': 0,
        'warning_count': 0,
        'issues': [],
        'issue_overflow_count': 0,
        'failed_symbols': [],
    }

    # Initialize components
    price_ingestor = PriceIngestor(config_path)
    news_enabled = stats['news_enabled']
    news_ingestor = NewsIngestor(config_path) if news_enabled else None
    feat_engineer = FeatureEngineer(config_path)
    model_manager = ModelManager(config_path)

    # For S&P 500 scale: avoid re-downloading full history for symbols that are already up-to-date.
    latest_market_date = None
    try:
        latest_market_date = price_ingestor.get_latest_market_date()
    except Exception:
        latest_market_date = None

    for ticker in tickers:
        try:
            logger.info(f"--- Processing {ticker} ---")

            # 1. Price Ingestion
            df_prices = None
            try:
                sym_last = price_ingestor.get_latest_date_for_symbol(ticker)
            except Exception:
                sym_last = None

            if latest_market_date and sym_last == latest_market_date:
                df_prices = None  # already current
            elif sym_last is None:
                # First time: prefer limited-window API if configured; otherwise download and prune later.
                src = str(getattr(price_ingestor, "price_source", "stooq") or "stooq").strip().lower()
                if src == "auto":
                    if price_ingestor.twelvedata_keys.keys():
                        src = "twelvedata"
                    elif price_ingestor.alphavantage_keys.keys():
                        src = "alphavantage"
                    else:
                        src = "stooq"
                if src == "twelvedata":
                    df_prices = price_ingestor.fetch_twelvedata_daily(ticker)
                    if df_prices is None or df_prices.empty:
                        if price_ingestor.alphavantage_keys.keys():
                            df_prices = price_ingestor.fetch_alphavantage_daily(ticker, outputsize="compact")
                        if df_prices is None or df_prices.empty:
                            df_prices = price_ingestor.fetch_stooq_data(ticker)
                elif src == "alphavantage":
                    df_prices = price_ingestor.fetch_alphavantage_daily(ticker, outputsize="compact")
                    if df_prices is None or df_prices.empty:
                        df_prices = price_ingestor.fetch_stooq_data(ticker)
                else:
                    df_prices = price_ingestor.fetch_stooq_data(ticker)
            else:
                # Incremental: fetch latest quote only.
                src = str(getattr(price_ingestor, "price_source", "stooq") or "stooq").strip().lower()
                if src == "auto":
                    if price_ingestor.twelvedata_keys.keys():
                        src = "twelvedata"
                    elif price_ingestor.alphavantage_keys.keys():
                        src = "alphavantage"
                    else:
                        src = "stooq"
                if src == "twelvedata":
                    df_prices = price_ingestor.fetch_twelvedata_daily(ticker, outputsize=5)
                    if (df_prices is None or df_prices.empty) and price_ingestor.alphavantage_keys.keys():
                        df_prices = price_ingestor.fetch_alphavantage_daily(ticker, outputsize="compact")
                elif src == "alphavantage":
                    df_prices = price_ingestor.fetch_alphavantage_daily(ticker, outputsize="compact")
                else:
                    df_prices = price_ingestor.fetch_stooq_latest(ticker)

            if df_prices is not None and not df_prices.empty:
                conn = sqlite3.connect(price_ingestor.db_path)
                price_ingestor._sqlite_upsert(
                    type('Table', (), {'name': 'prices'}),
                    conn,
                    df_prices.columns.tolist(),
                    df_prices.values.tolist()
                )
                conn.close()

            # 2. News (Optional)
            if news_ingestor is not None:
                news_ingestor.fetch_and_store_news(ticker)

            # 3/4. Features + Training (only when retraining is needed)
            needs_retrain = True
            try:
                needs_retrain = model_manager._should_retrain(str(ticker))
            except Exception:
                needs_retrain = True

            model_id = None
            if needs_retrain:
                feat_engineer.generate_and_save(ticker)
                model_id = model_manager.train_ols(ticker)
            if model_id:
                stats['models_trained'] += 1

            stats['tickers_processed'] += 1

            # Rate limiting for APIs
            time.sleep(0.3 if sym_last is not None else 1.0)

        except Exception as e:
            stats['tickers_failed'] += 1
            logger.error(f"Failed to process {ticker}: {e}")
            if len(stats.get("failed_symbols", [])) < 25:
                stats["failed_symbols"].append(str(ticker))
            _record_pipeline_issue(stats, "ERROR", f"Ticker {ticker}", str(e))
            continue

    if news_ingestor is not None:
        stats['llm_status'] = news_ingestor.get_llm_status()

    signal_payload = []
    try:
        signal_payload = build_signal_snapshot(config_path, tickers)
        logger.info("Backtest signals snapshot generated.")
    except Exception as exc:
        logger.error(f"Failed to generate backtest signals snapshot: {exc}")
        _record_pipeline_issue(stats, "WARNING", "Backtest Signal Snapshot", str(exc))

    logger.info("Pipeline completed. Running backtest...")
    backtester = DailyBacktester(config_path)
    email_sent = False
    _finalize_pipeline_health(stats)
    result = backtester.run_daily_test(
        pipeline_stats=stats,
        backtest_signals=signal_payload
    )
    if result:
        summary, unrealized, closed_positions, email_sent = result
        logger.info(f"Daily Summary: {summary}")
    else:
        logger.warning("Backtest skipped (no predictions or data).")

    # Meta-Learning: Analyze past trades and update adjustment factors
    logger.info("Running meta-learner analysis...")
    from meta_learner import MetaLearner
    meta = MetaLearner(config_path)
    meta.analyze_past_trades()
    logger.info("Meta-learner state updated for next cycle.")

    # Enforce storage limits (prune/vacuum) to keep disk usage low for large universes.
    try:
        from storage_policy import apply_storage_policy
        sres = apply_storage_policy(config_path)
        logger.info("Storage policy applied: %s", sres)
    except Exception as exc:
        logger.warning("Storage policy failed: %s", exc)
    return email_sent


def run_daily_job(config_path=None):
    """
    Fast daily job for large universes (S&P 500):
    - Update latest prices (API if available, otherwise Stooq fallback)
    - Fetch news/LLM sentiment only for currently-held symbols (keeps runtime bounded)
    - Run daily position-tracking backtest and send emails
    - Apply storage policy (retention/vacuum/prune)
    """
    config_path = _get_config_path(config_path)
    config = _load_config(config_path)
    base_dir = os.path.dirname(os.path.abspath(config_path))
    universe_path = _resolve_path(base_dir, config['universe']['source'])

    universe_df = pd.read_csv(universe_path)
    tickers = [str(t).strip().upper() for t in universe_df['ticker'].tolist() if str(t).strip()]

    price_ingestor = PriceIngestor(config_path)
    providers = (config.get("data", {}).get("providers", {}) or {})
    td = providers.get("twelvedata", {}) if isinstance(providers, dict) else {}
    assumed_rpm = int(td.get("assumed_requests_per_minute_per_key", 5) or 5)
    key_count = len(price_ingestor.twelvedata_keys.keys())
    # Throttle globally so per-key usage stays under free-tier limits.
    # overall_rps ~= keys * rpm / 60
    overall_rps = (key_count * assumed_rpm) / 60.0 if key_count > 0 else 0.0
    sleep_s = 1.0 / overall_rps if overall_rps > 0 else 0.6
    sleep_s = max(0.2, min(2.0, sleep_s))

    latest_market_date = None
    try:
        latest_market_date = price_ingestor.get_latest_market_date()
    except Exception:
        latest_market_date = None

    results_dir = _resolve_path(
        base_dir,
        (config.get("output", {}) or {}).get("results_dir", "./results")
    )

    # Determine the provider's latest available market date using a single "sentinel" symbol.
    # This avoids a deadlock where the scheduler won't run because DB market date is stale.
    src = str(getattr(price_ingestor, "price_source", "stooq") or "stooq").strip().lower()
    if src == "auto":
        if price_ingestor.twelvedata_keys.keys():
            src = "twelvedata"
        elif price_ingestor.alphavantage_keys.keys():
            src = "alphavantage"
        else:
            src = "stooq"

    pipeline_stats = {
        'start_time': get_sgt_now().isoformat(),
        'steps': [],
        'tickers_total': len(tickers),
        'tickers_processed': 0,
        'tickers_failed': 0,
        'news_fetched': 0,
        'news_enabled': bool(config.get('news', {}).get('enabled', False)),
        'llm_status': None,
        'error_count': 0,
        'warning_count': 0,
        'issues': [],
        'issue_overflow_count': 0,
        'failed_symbols': [],
    }
    
    def log_step(name, status, details=None):
        pipeline_stats['steps'].append({
            'time': get_sgt_now().strftime("%H:%M:%S"),
            'step': name,
            'status': status,
            'details': details
        })
        logger.info(f"STEP: {name} | STATUS: {status} | {details or ''}")
        status_u = str(status or "").strip().upper()
        if status_u == "FAILED":
            _record_pipeline_issue(pipeline_stats, "ERROR", name, details or "step failed")
        elif status_u == "WARNING":
            _record_pipeline_issue(pipeline_stats, "WARNING", name, details or "step warning")

    # One-time recovery for cloud cache misses: seed only if tables are empty.
    log_step("State Recovery", "Started")
    try:
        recovery = recover_positions_from_seed(config_path)
        if recovery.get("recovered_total", 0) > 0:
            log_step(
                "State Recovery",
                "Completed",
                f"Recovered core={recovery.get('recovered_core', 0)} ai={recovery.get('recovered_ai', 0)}",
            )
        else:
            log_step("State Recovery", "Skipped", recovery.get("reason", "no_recovery_needed"))
    except Exception as exc:
        log_step("State Recovery", "Failed", str(exc))

    # Safety pass: if historical open positions exceed account capital,
    # shrink quantities before today's strategy logic runs.
    log_step("Position Sanity", "Started")
    try:
        sanity = enforce_position_cap(config_path)
        adjusted_total = int(sanity.get("adjusted_total", 0) or 0)
        if adjusted_total > 0:
            details = []
            core_tbl = (sanity.get("tables", {}) or {}).get("positions", {}) or {}
            ai_tbl = (sanity.get("tables", {}) or {}).get("positions_ai", {}) or {}
            if core_tbl.get("adjusted"):
                details.append(f"core={core_tbl.get('adjusted')}")
            if ai_tbl.get("adjusted"):
                details.append(f"ai={ai_tbl.get('adjusted')}")
            log_step("Position Sanity", "Completed", "Adjusted " + ", ".join(details))
        else:
            log_step("Position Sanity", "Skipped", "within_cap")
    except Exception as exc:
        log_step("Position Sanity", "Failed", str(exc))

    log_step("Market Check", "Started", f"Source: {src}")

    sentinel = "AAPL" if "AAPL" in tickers else (tickers[0] if tickers else "AAPL")
    provider_latest_date = None
    try:
        if src == "twelvedata":
            df = price_ingestor.fetch_twelvedata_daily(sentinel, outputsize=5)
            if df is not None and not df.empty and "date" in df.columns:
                provider_latest_date = str(df["date"].max())
        elif src == "alphavantage":
            df = price_ingestor.fetch_alphavantage_daily(sentinel, outputsize="compact")
            if df is not None and not df.empty and "date" in df.columns:
                provider_latest_date = str(df["date"].max())
        else:
            df = price_ingestor.fetch_stooq_latest(sentinel)
            if df is not None and not df.empty and "date" in df.columns:
                provider_latest_date = str(df["date"].max())
    except Exception:
        provider_latest_date = None

    # If no new provider data is available AND we've already emailed for the latest DB market date,
    # do nothing (prevents "weekend reruns" from opening/closing positions or resending emails).
    if latest_market_date and provider_latest_date and provider_latest_date <= latest_market_date:
        marker = os.path.join(results_dir, f"email_sent_{str(latest_market_date).replace('-', '')}.ok")
        if os.path.exists(marker):
            logger.info(
                "No new market data (provider_latest=%s, db_latest=%s) and email already sent. Skipping run.",
                provider_latest_date,
                latest_market_date,
            )
            return True

    model_manager = ModelManager(config_path)

    # Update prices for all tickers (incremental) without downloading full history repeatedly.
    log_step("Price Ingestion", "Started", f"Processing {len(tickers)} symbols...")
    for t in tickers:
        try:
            sym_last = None
            try:
                sym_last = price_ingestor.get_latest_date_for_symbol(t)
            except Exception:
                sym_last = None

            df_prices = None
            src = str(getattr(price_ingestor, "price_source", "stooq") or "stooq").strip().lower()
            if src == "auto":
                if price_ingestor.twelvedata_keys.keys():
                    src = "twelvedata"
                elif price_ingestor.alphavantage_keys.keys():
                    src = "alphavantage"
                else:
                    src = "stooq"

            if sym_last is None:
                # First time: limited window.
                if src == "twelvedata":
                    df_prices = price_ingestor.fetch_twelvedata_daily(t)
                    if df_prices is None or df_prices.empty:
                        if price_ingestor.alphavantage_keys.keys():
                            df_prices = price_ingestor.fetch_alphavantage_daily(t, outputsize="compact")
                        if df_prices is None or df_prices.empty:
                            df_prices = price_ingestor.fetch_stooq_data(t)
                elif src == "alphavantage":
                    df_prices = price_ingestor.fetch_alphavantage_daily(t, outputsize="compact")
                    if df_prices is None or df_prices.empty:
                        df_prices = price_ingestor.fetch_stooq_data(t)
                else:
                    df_prices = price_ingestor.fetch_stooq_data(t)
            else:
                # Incremental
                if provider_latest_date and sym_last == provider_latest_date:
                    df_prices = None
                elif src == "twelvedata":
                    df_prices = price_ingestor.fetch_twelvedata_daily(t, outputsize=5)
                    if (df_prices is None or df_prices.empty) and price_ingestor.alphavantage_keys.keys():
                        df_prices = price_ingestor.fetch_alphavantage_daily(t, outputsize="compact")
                elif src == "alphavantage":
                    df_prices = price_ingestor.fetch_alphavantage_daily(t, outputsize="compact")
                else:
                    df_prices = price_ingestor.fetch_stooq_latest(t)

            if df_prices is not None and not df_prices.empty:
                conn = sqlite3.connect(price_ingestor.db_path)
                try:
                    price_ingestor._sqlite_upsert(
                        type('Table', (), {'name': 'prices'}),
                        conn,
                        df_prices.columns.tolist(),
                        df_prices.values.tolist()
                    )
                finally:
                    conn.close()
            
            pipeline_stats['tickers_processed'] += 1
            
            # Train if model is missing or stale
            model_manager.train_ols(t)

            time.sleep(sleep_s)
        except Exception as exc:
            pipeline_stats['tickers_failed'] += 1
            logger.warning(f"Failed to ingest {t}: {exc}")
            if len(pipeline_stats.get("failed_symbols", [])) < 25:
                pipeline_stats["failed_symbols"].append(str(t))
            _record_pipeline_issue(pipeline_stats, "ERROR", f"Price Ingestion:{t}", str(exc))
            continue

    log_step("Price Ingestion", "Completed", f"Success: {pipeline_stats['tickers_processed']}, Failed: {pipeline_stats['tickers_failed']}")

    # News/LLM sentiment only for held symbols (keeps runtime within budget).
    news_enabled = pipeline_stats['news_enabled']
    if news_enabled:
        log_step("News Ingestion", "Started")
        try:
            news_ingestor = NewsIngestor(config_path)
            core_tracker = PositionTracker(config_path, table_name="positions")
            ai_tracker = PositionTracker(config_path, table_name="positions_ai")
            held = set()
            for df in [core_tracker.get_open_positions(), ai_tracker.get_open_positions()]:
                if df is not None and hasattr(df, "empty") and not df.empty:
                    held |= set([str(s).strip().upper() for s in df["symbol"].tolist() if str(s).strip()])
            
            for sym in sorted(list(held))[:50]:
                try:
                    news_ingestor.fetch_and_store_news(sym)
                    pipeline_stats['news_fetched'] += 1
                except Exception as exc:
                    _record_pipeline_issue(pipeline_stats, "WARNING", f"News Ingestion:{sym}", str(exc))
                    continue
            try:
                pipeline_stats['llm_status'] = news_ingestor.get_llm_status()
                llm_status = pipeline_stats.get('llm_status') if isinstance(pipeline_stats, dict) else None
                if isinstance(llm_status, dict) and int(llm_status.get("errors", 0) or 0) > 0:
                    _record_pipeline_issue(
                        pipeline_stats,
                        "WARNING",
                        "News Sentiment",
                        llm_status.get("last_error") or f"errors={llm_status.get('errors')}",
                    )
            except Exception as exc:
                _record_pipeline_issue(pipeline_stats, "WARNING", "News Sentiment", str(exc))
            log_step("News Ingestion", "Completed", f"Fetched for {pipeline_stats['news_fetched']} symbols")
        except Exception as exc:
            log_step("News Ingestion", "Failed", str(exc))
    else:
        log_step("News Ingestion", "Skipped", "Disabled in config")

    # Run daily backtest + emails
    log_step("Backtest & Strategy", "Started")
    backtester = DailyBacktester(config_path)
    _finalize_pipeline_health(pipeline_stats)
    result = backtester.run_daily_test(pipeline_stats=pipeline_stats)
    email_sent = bool(result and result[-1])
    
    if email_sent:
        log_step("Backtest & Strategy", "Completed", "Report sent")
    else:
        log_step("Backtest & Strategy", "Warning", "Email failed or no signals")

    # Enforce storage limits
    try:
        from storage_policy import apply_storage_policy
        apply_storage_policy(config_path)
    except Exception:
        pass

    return email_sent


if __name__ == "__main__":
    import sys
    mode = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
    if mode in {"full", "pipeline"}:
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        success = run_full_pipeline(limit=limit)
    else:
        # Default is the fast daily job (scheduler-friendly).
        success = run_daily_job()
    sys.exit(0 if success else 1)
