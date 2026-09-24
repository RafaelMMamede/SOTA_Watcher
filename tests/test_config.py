import tempfile
import unittest
from pathlib import Path

from utils.config import load_config


class ConfigTests(unittest.TestCase):
    def test_duplicate_yaml_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "screening:\n"
                "  enabled: true\n"
                "  max_papers_per_run: 2\n"
                "  max_papers_per_run: null\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "Duplicate YAML key 'max_papers_per_run'",
            ):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
