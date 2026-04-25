import sqlite3
import tempfile
import unittest
from pathlib import Path

from ai_manager_memory import AIManagerMemory
from distilled_trade_client import DistilledTradeClient


class AISharedMemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "trading_bot.db"
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE positions_ai (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                entry_date TEXT,
                exit_date TEXT,
                realized_pnl REAL,
                realized_pnl_dollars REAL,
                status TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO positions_ai (symbol, side, entry_date, exit_date, realized_pnl, realized_pnl_dollars, status) VALUES (?, ?, ?, ?, ?, ?, 'CLOSED')",
            [
                ("AAA", "LONG", "2026-04-01", "2026-04-05", 0.08, 800.0),
                ("AAA", "LONG", "2026-04-08", "2026-04-12", 0.04, 400.0),
                ("BBB", "SHORT", "2026-04-02", "2026-04-06", 0.06, 600.0),
                ("BBB", "SHORT", "2026-04-09", "2026-04-13", -0.03, -300.0),
            ],
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_manager_memory_tracks_runs_and_symbol_bias(self):
        memory = AIManagerMemory(str(self.db_path), lookback_days=365)
        memory.record_run(
            run_date="2026-04-25",
            backend_selected="trained_model_http",
            requested_mode="auto",
            model_used="quant-trained-trading-model",
            ok=True,
            candidates_seen=1000,
            candidates_scored=120,
            target_positions=10,
            notes={"router_reason": "full_model_ready"},
        )
        memory.record_trade_plan(
            run_date="2026-04-25",
            backend_selected="trained_model_http",
            trades=[{"symbol": "AAA", "side": "LONG", "weight": 0.5, "confidence": 0.9, "score": 1.0, "label": "BUY", "reason": "test"}],
        )

        ctx = memory.build_context()
        self.assertEqual(ctx["last_backend"], "trained_model_http")
        self.assertIn("trained_model_http", ctx["success_rate_by_backend"])

        bias = memory.symbol_side_bias()
        self.assertGreater(bias[("AAA", "LONG")]["bias"], 0.0)
        self.assertGreater(bias[("BBB", "SHORT")]["confidence"], 0.0)

    def test_distilled_client_uses_shared_memory_bias(self):
        memory = AIManagerMemory(str(self.db_path), lookback_days=365)
        client = DistilledTradeClient(
            {
                "data": {"cache_path": str(self.db_path)},
                "ai_trading": {"runtime_router": {"distilled_model_name": "distilled-feature-manager", "memory_lookback_days": 365}},
            },
            manager_memory=memory,
        )
        preds = client.predict_candidates(
            [
                {
                    "symbol": "AAA",
                    "return_1d": 0.01,
                    "return_5d": 0.03,
                    "return_10d": 0.05,
                    "dist_ma_20": 0.04,
                    "dist_ma_50": 0.05,
                    "rsi_14": 61,
                    "volume_ratio": 1.2,
                    "news_count_7d": 3,
                    "news_sentiment_7d": 0.4,
                },
                {
                    "symbol": "BBB",
                    "return_1d": -0.01,
                    "return_5d": -0.04,
                    "return_10d": -0.06,
                    "dist_ma_20": -0.05,
                    "dist_ma_50": -0.06,
                    "rsi_14": 39,
                    "volume_ratio": 1.1,
                    "news_count_7d": 2,
                    "news_sentiment_7d": -0.2,
                },
            ]
        )
        self.assertEqual(preds[0]["label"], "STRONG_BUY")
        self.assertIn("long", preds[0]["reason"].lower())
        self.assertIn(preds[1]["label"], {"SELL", "STRONG_SELL"})
        self.assertIn("short", preds[1]["reason"].lower())


if __name__ == "__main__":
    unittest.main()
