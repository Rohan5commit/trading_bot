import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from main import DailyBacktester


class _DummyMetaLearner:
    def __init__(self, *_args, **_kwargs):
        pass

    def analyze_past_trades(self, lookback_days=30):
        return None

    def get_confidence_adjustments(self, rank_df):
        out = rank_df.copy()
        if "adjusted_score" not in out.columns:
            out["adjusted_score"] = out["predicted_return"]
        if "penalty" not in out.columns:
            out["penalty"] = 1.0
        return out

    def get_daily_insights(self):
        return "meta stub"

    def get_exit_cooldown_symbols(self, as_of_date=None):
        return set()


class _DummyNotifier:
    calls = []

    def __init__(self, *_args, **_kwargs):
        pass

    def send_daily_report(self, **kwargs):
        self.__class__.calls.append(kwargs)
        return True


class AutonomousAITest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self.results_dir = self.tmp_path / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.tmp_path / "trading_bot.db"
        self.universe_path = self.tmp_path / "universe.csv"
        self.universe_path.write_text("ticker\nB\nD\nE\n")

        repo_root = Path(__file__).resolve().parents[1]
        base_config = yaml.safe_load((repo_root / "config.yaml").read_text())
        base_config["data"]["cache_path"] = str(self.db_path)
        base_config["output"]["results_dir"] = str(self.results_dir)
        base_config["universe"]["source"] = str(self.universe_path)
        base_config["ai_trading"]["prompt_candidates_limit"] = 2
        base_config["ai_trading"]["disallow_core_overlap"] = False
        base_config["ai_trading"]["position_management_mode"] = "autonomous_rebalance"
        base_config["ai_trading"]["min_trade_dollars"] = 500
        base_config["state_recovery"]["enabled"] = False
        self.config_path = self.tmp_path / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(base_config))

        self.test_date = pd.Timestamp("2026-04-24")
        self.signal_date = pd.Timestamp("2026-04-23")
        self._seed_prices(["B", "C", "D", "E"], start="2026-01-05", end="2026-04-24")

        backtester = DailyBacktester(str(self.config_path))
        backtester.ai_tracker.open_position(
            symbol="C",
            entry_date="2026-04-23",
            entry_price=100.0,
            quantity=1000.0,
            side="LONG",
            target_price=100.0,
            decision_label="BUY",
            decision_confidence=0.7,
            decision_reason="legacy position",
            last_decision_date="2026-04-23",
        )

    def tearDown(self):
        self._tmpdir.cleanup()

    def _seed_prices(self, symbols, start, end):
        dates = pd.date_range(start=start, end=end, freq="B")
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                datetime TEXT,
                title TEXT,
                url TEXT,
                sentiment_score REAL
            )
            """
        )
        for symbol in symbols:
            base = {"B": 110.0, "C": 100.0, "D": 90.0, "E": 80.0}[symbol]
            for idx, date in enumerate(dates):
                px = base + (idx * 0.25)
                conn.execute(
                    "INSERT INTO prices (symbol, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        symbol,
                        date.strftime("%Y-%m-%d"),
                        px,
                        px + 1.5,
                        px - 1.5,
                        px + 0.5,
                        100000 + idx,
                    ),
                )
        conn.commit()
        conn.close()

    def test_ai_rebalances_without_cash_and_keeps_open_positions_in_scope(self):
        _DummyNotifier.calls = []

        def fake_rankings(self, symbols, signal_date, conn):
            rows = []
            for idx, symbol in enumerate(symbols, start=1):
                rows.append({"symbol": symbol, "predicted_return": 0.02 * (len(symbols) - idx + 1)})
            return rows

        def fake_ai_decisions(config, candidates, max_positions=10, allow_shorts=True, max_shorts=5):
            candidate_symbols = [str(item.get("symbol") or "").strip().upper() for item in candidates]
            self.assertIn("C", candidate_symbols, "Open AI position must always be analyzed.")
            self.assertLessEqual(len(candidates), 2, "Prompt limit should still cap non-held candidates.")
            target_symbol = next(symbol for symbol in candidate_symbols if symbol != "C")
            return (
                [
                    {
                        "symbol": target_symbol,
                        "side": "LONG",
                        "weight": 0.96,
                        "reason": "rotate into stronger setup",
                        "label": "STRONG_BUY",
                        "confidence": 0.91,
                        "score": 2.0,
                    }
                ],
                {
                    "ok": True,
                    "error": None,
                    "model_used": "unit-test-model",
                    "decision_engine": "trained_model",
                    "candidates_seen": len(candidates),
                    "candidates_scored": 1,
                },
            )

        with (
            patch.dict("os.environ", {"DISABLE_CORE_TRADING": "1"}, clear=False),
            patch("meta_learner.MetaLearner", _DummyMetaLearner),
            patch("email_notifier.EmailNotifier", _DummyNotifier),
            patch.object(DailyBacktester, "get_predictions_for_date_bulk", fake_rankings),
            patch("llm_trader.propose_trades_with_llm", fake_ai_decisions),
        ):
            backtester = DailyBacktester(str(self.config_path))
            report, unrealized, closed_positions, email_sent = backtester.run_daily_test(self.test_date)

        self.assertIsNone(report)
        self.assertTrue(email_sent)
        self.assertEqual(len(_DummyNotifier.calls), 1)
        self.assertEqual(_DummyNotifier.calls[0]["subject_tag"], "AI")

        tracker = DailyBacktester(str(self.config_path)).ai_tracker
        open_positions = tracker.get_open_positions()
        self.assertEqual(len(open_positions), 1)
        remaining_symbol = str(open_positions.iloc[0]["symbol"]).strip().upper()
        self.assertNotEqual(remaining_symbol, "C")
        self.assertAlmostEqual(float(open_positions.iloc[0]["target_price"]), float(open_positions.iloc[0]["entry_price"]), places=6)
        self.assertEqual(str(open_positions.iloc[0]["decision_label"]).upper(), "STRONG_BUY")

        summary = tracker.get_portfolio_summary()
        self.assertEqual(int(summary["closed_positions"]), 1)
        self.assertEqual(int(summary["open_positions"]), 1)

        ai_report_path = self.results_dir / f"daily_report_ai_{self.test_date.strftime('%Y%m%d')}.csv"
        self.assertTrue(ai_report_path.exists())
        ai_report = pd.read_csv(ai_report_path).iloc[0].to_dict()
        self.assertEqual(ai_report["ai_position_management_mode"], "autonomous_rebalance")
        self.assertEqual(int(ai_report["positions_closed_by_ai"]), 1)
        self.assertEqual(int(ai_report["new_positions_opened"]), 1)
        self.assertTrue(bool(ai_report["ai_llm_ok"]))

        ai_unrealized_path = self.results_dir / f"unrealized_ai_{self.test_date.strftime('%Y%m%d')}.csv"
        self.assertTrue(ai_unrealized_path.exists())
        ai_unrealized = pd.read_csv(ai_unrealized_path)
        self.assertIn("decision_label", ai_unrealized.columns)
        self.assertIn("decision_reason", ai_unrealized.columns)
        self.assertIn("current_price_date", ai_unrealized.columns)


if __name__ == "__main__":
    unittest.main()
