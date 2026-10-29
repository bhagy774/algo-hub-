import importlib.util
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("downloader", Path(__file__).with_name("jarvis_historical_downloader.py"))
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class DownloaderTest(unittest.TestCase):
    def test_parse_utc_and_interval(self):
        self.assertEqual(mod.parse_utc("2024-01-01").tzinfo, timezone.utc)
        self.assertEqual(mod.interval_close_ms(1_000, "1m"), 61_000)

    def test_download_normalizes_and_keeps_completed_candles(self):
        pages = [
            [[0, "10", "12", "9", "11", "100"], [60_000, "11", "13", "10", "12", "101"]],
            []
        ]
        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls.fromtimestamp(180, timezone.utc)
        with patch.object(mod, "fetch_page", side_effect=lambda *args, **kwargs: pages.pop(0)), patch.object(mod, "datetime", FixedDateTime):
            frame = mod.download("BTCUSDT", "1m", datetime.fromtimestamp(0, timezone.utc), datetime.fromtimestamp(120, timezone.utc))
        self.assertEqual(list(frame.columns), mod.COLUMNS)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame["close"].tolist(), [11, 12])


if __name__ == "__main__":
    unittest.main()
