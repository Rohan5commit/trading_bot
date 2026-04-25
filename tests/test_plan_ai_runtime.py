import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import sys

from ai_manager_memory import AIManagerMemory
from quant_platform.scripts.plan_ai_runtime import choose_runtime, write_github_output


class PlanAIRuntimeTest(unittest.TestCase):
    def test_write_github_output_escapes_multiline_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "github_output.txt"
            write_github_output(
                str(output_path),
                {
                    "runtime_mode": "distilled_local",
                    "selected_backend": "distilled_local",
                    "selected_compute_name": "",
                    "selected_disk_gb": 0,
                    "reason": "line1\nline2",
                },
            )
            text = output_path.read_text()
            self.assertIn("reason<<__CODEx_EOF__", text)
            self.assertIn("line1\nline2", text)

    def test_choose_runtime_respects_full_runtime_blockers(self):
        fake_report = {
            "user_balance": 25.0,
            "project_balance": 0.0,
            "completed_signup": True,
            "feature_flags": {"persistentDisk": False},
        }
        fake_module = ModuleType("lightning_account_preflight")
        fake_module.build_preflight_report = lambda: fake_report
        with patch("quant_platform.scripts.plan_ai_runtime._load_router_config", return_value={}):
            with patch.dict(sys.modules, {"lightning_account_preflight": fake_module}):
                payload = choose_runtime()
        self.assertEqual(payload["runtime_mode"], "distilled_local")
        self.assertIn("full_runtime_blocked", payload["reason"])

    def test_ai_manager_memory_uses_config_base_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            db_path = base_dir / "data" / "trading_bot.db"
            memory = AIManagerMemory.from_config(
                {
                    "_config_base_dir": str(base_dir),
                    "data": {"cache_path": "data/trading_bot.db"},
                    "ai_trading": {"runtime_router": {"memory_lookback_days": 30}},
                }
            )
            self.assertEqual(Path(memory.db_path), db_path)


if __name__ == "__main__":
    unittest.main()
