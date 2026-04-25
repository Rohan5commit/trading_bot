import unittest
from pathlib import Path


class ConfigSecretHygieneTest(unittest.TestCase):
    def test_tracked_provider_keys_are_empty(self):
        text = (Path(__file__).resolve().parents[1] / "config.yaml").read_text()
        self.assertIn("api_keys: []", text)


if __name__ == "__main__":
    unittest.main()
