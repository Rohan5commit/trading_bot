import tempfile
import unittest
from pathlib import Path

from quant_platform.scripts.prepare_ai_only_retry import clear_ai_retry_state


class PrepareAIOnlyRetryTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tmpdir.name) / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "email_sent_20260425.ok",
            "email_sent_ai_20260425.ok",
            "email_sent_core_20260425.ok",
            "daily_report_ai_20260425.csv",
            "unrealized_ai_20260425.csv",
            "trades_ai_20260425.csv",
        ):
            (self.results_dir / name).write_text("x")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_clear_ai_retry_state_keeps_core_marker_only(self):
        payload = clear_ai_retry_state(self.results_dir)
        self.assertIn("email_sent_core_20260425.ok", payload["kept_markers"])
        self.assertFalse((self.results_dir / "email_sent_20260425.ok").exists())
        self.assertFalse((self.results_dir / "email_sent_ai_20260425.ok").exists())
        self.assertTrue((self.results_dir / "email_sent_core_20260425.ok").exists())
        self.assertFalse((self.results_dir / "daily_report_ai_20260425.csv").exists())
        self.assertFalse((self.results_dir / "unrealized_ai_20260425.csv").exists())
        self.assertFalse((self.results_dir / "trades_ai_20260425.csv").exists())


if __name__ == "__main__":
    unittest.main()
