import sqlite3
import tempfile
import unittest
from pathlib import Path

import yaml

from positions import PositionTracker


class PositionTrackerAITest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self.config_path = self.tmp_path / "config.yaml"
        self.db_path = self.tmp_path / "trading_bot.db"
        config = {
            "data": {"cache_path": str(self.db_path)},
            "trading": {"take_profit_pct": 0.03},
        }
        self.config_path.write_text(yaml.safe_dump(config))

    def tearDown(self):
        self._tmpdir.cleanup()

    def _tracker(self):
        return PositionTracker(str(self.config_path), table_name="positions_ai")

    def _seed_price(self, symbol: str, date: str, open_px: float, high_px: float, low_px: float, close_px: float):
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
            "INSERT INTO prices (symbol, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (symbol, date, open_px, high_px, low_px, close_px, 1000.0),
        )
        conn.commit()
        conn.close()

    def test_ai_position_metadata_and_unrealized_snapshot(self):
        tracker = self._tracker()
        self._seed_price("AAPL", "2026-04-23", 200.0, 205.0, 198.0, 202.5)

        tracker.open_position(
            symbol="AAPL",
            entry_date="2026-04-23",
            entry_price=200.0,
            quantity=10.0,
            side="LONG",
            target_price=200.0,
            decision_label="BUY",
            decision_confidence=0.77,
            decision_reason="positive momentum bias",
            last_decision_date="2026-04-23",
        )
        tracker.update_position_decision(
            "AAPL",
            "2026-04-24",
            decision_label="STRONG_BUY",
            decision_confidence=0.88,
            decision_reason="momentum still strong",
            target_price=200.0,
        )

        unrealized = tracker.get_unrealized_pnl()
        self.assertFalse(unrealized.empty)
        row = unrealized.iloc[0]
        self.assertEqual(row["symbol"], "AAPL")
        self.assertEqual(row["current_price"], 202.5)
        self.assertEqual(row["current_price_date"], "2026-04-23")
        self.assertEqual(row["decision_label"], "STRONG_BUY")
        self.assertAlmostEqual(float(row["decision_confidence"]), 0.88, places=2)
        self.assertEqual(row["decision_reason"], "momentum still strong")
        self.assertEqual(row["last_decision_date"], "2026-04-24")

    def test_close_position_is_side_aware_for_short(self):
        tracker = self._tracker()
        tracker.open_position(
            symbol="TSLA",
            entry_date="2026-04-23",
            entry_price=300.0,
            quantity=5.0,
            side="SHORT",
            target_price=300.0,
            decision_label="SELL",
            decision_confidence=0.71,
            decision_reason="negative momentum bias",
            last_decision_date="2026-04-23",
        )

        closed = tracker.close_position(
            symbol="TSLA",
            exit_date="2026-04-24",
            exit_price=270.0,
            reason="AI rotation: covered after improved signal",
        )

        self.assertIsNotNone(closed)
        self.assertEqual(closed["side"], "SHORT")
        self.assertAlmostEqual(float(closed["realized_pnl"]), 0.10, places=4)
        self.assertAlmostEqual(float(closed["realized_pnl_dollars"]), 150.0, places=2)

        open_positions = tracker.get_open_positions()
        self.assertTrue(open_positions.empty)
        summary = tracker.get_portfolio_summary()
        self.assertEqual(summary["closed_positions"], 1)
        self.assertAlmostEqual(float(summary["total_realized_pnl_dollars"]), 150.0, places=2)


if __name__ == "__main__":
    unittest.main()
