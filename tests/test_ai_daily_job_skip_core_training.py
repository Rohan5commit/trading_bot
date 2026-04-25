import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from main import run_daily_job


class _DummyKeys:
    def keys(self):
        return []


class _DummyPriceIngestor:
    def __init__(self, config_path):
        self.config_path = config_path
        self.db_path = str(Path(config_path).with_suffix(".db"))
        self.price_source = "stooq"
        self.twelvedata_keys = _DummyKeys()
        self.alphavantage_keys = _DummyKeys()

    def get_latest_market_date(self):
        return None

    def get_latest_date_for_symbol(self, _symbol):
        return None

    def fetch_stooq_data(self, symbol):
        return pd.DataFrame(
            [
                {
                    "symbol": str(symbol).strip().upper(),
                    "date": "2026-04-24",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1000,
                }
            ]
        )

    fetch_stooq_latest = fetch_stooq_data

    def _sqlite_upsert(self, *_args, **_kwargs):
        return None


class _RaisingModelManager:
    def __init__(self, *_args, **_kwargs):
        raise AssertionError("AI-only daily job should not construct core ModelManager.")


class _DummyBacktester:
    def __init__(self, *_args, **_kwargs):
        pass

    def run_daily_test(self, pipeline_stats=None):
        return ({}, pd.DataFrame(), [], True)


class AIDailyJobSkipsCoreTrainingTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self.results_dir = self.tmp_path / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.universe_path = self.tmp_path / "universe.csv"
        self.universe_path.write_text("ticker\nAAPL\n")

        repo_root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((repo_root / "config.yaml").read_text())
        config["output"]["results_dir"] = str(self.results_dir)
        config["universe"]["source"] = str(self.universe_path)
        config["news"]["enabled"] = False
        config["ai_trading"]["enabled"] = True
        self.config_path = self.tmp_path / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(config))

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_ai_only_daily_job_skips_core_model_training(self):
        with (
            patch.dict("os.environ", {"DISABLE_CORE_TRADING": "1"}, clear=False),
            patch("main.PriceIngestor", _DummyPriceIngestor),
            patch("main.ModelManager", _RaisingModelManager),
            patch("main._get_open_position_symbols", lambda _config_path: []),
            patch("main.DailyBacktester", _DummyBacktester),
            patch("main.time.sleep", lambda *_args, **_kwargs: None),
            patch("storage_policy.apply_storage_policy", lambda _config_path: None),
        ):
            self.assertTrue(run_daily_job(str(self.config_path)))


if __name__ == "__main__":
    unittest.main()
