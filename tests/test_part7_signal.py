import io
import time
import unittest
from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

from jarvis_dashboard import render_dashboard
from part7_signal import DEFAULT_TIMEFRAMES, aggregate_results, analyze_timeframe
from binance_timeframes import candle_open_time


SYMBOL = "ETHUSDT"


def candles(*, timestamp=None, timeframe="1m", extreme=False, symbol=SYMBOL, rows=60):
    """Build deterministic offline fixtures on exact native Binance boundaries."""
    close = np.full(rows, 100.0)
    high = close + 0.2
    low = close - 0.2
    if extreme:
        high[-8:] = close[-8:] + 5.0
        low[-8:] = close[-8:] - 5.0
    frame = pd.DataFrame({
        "open": close.copy(), "high": high, "low": low,
        "close": close.copy(), "volume": np.full(rows, 1000.0),
    })
    if timestamp is None:
        now = pd.Timestamp.now(tz="UTC").timestamp()
        current_open = candle_open_time(timeframe, now)
        last_closed = candle_open_time(timeframe, current_open - 1)
    else:
        stamp = pd.Timestamp(timestamp)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        last_closed = candle_open_time(timeframe, stamp.timestamp())
    opens = [int(last_closed)]
    for _ in range(rows - 1):
        opens.append(candle_open_time(timeframe, opens[-1] - 1))
    frame.index = pd.to_datetime(list(reversed(opens)), unit="s", utc=True)
    frame.attrs["symbol"] = symbol
    return frame


class Part7SignalContracts(unittest.TestCase):
    def live_context(self, timeframe="1m", symbol=SYMBOL):
        return {"selected_symbol": symbol, "timeframe": timeframe,
                "is_backtest_mode": False, "shared_candle_source": "shared_exchange"}

    def test_routes_each_configured_native_timeframe_with_identity(self):
        for timeframe in DEFAULT_TIMEFRAMES:
            result = analyze_timeframe(candles(timeframe=timeframe), symbol=SYMBOL,
                                       timeframe=timeframe,
                                       context=self.live_context(timeframe))
            self.assertEqual(result["symbol"], SYMBOL)
            self.assertEqual(result["timeframe"], timeframe)
            self.assertEqual(result["signal_identity"], f"part7:{SYMBOL}:{timeframe}")
            self.assertEqual(result["computation_backend"], "stdlib_cpu")
            self.assertEqual(result["accelerator"], "cpu")

    def test_wrong_or_missing_symbol_identity_blocks(self):
        wrong = analyze_timeframe(candles(symbol="BTCUSDT"), symbol=SYMBOL,
                                  timeframe="1m", context=self.live_context())
        self.assertEqual(wrong["status"], "symbol_mismatch")
        self.assertTrue(wrong["entry_blocked"])
        missing = candles()
        missing.attrs = {}
        result = analyze_timeframe(missing, symbol=SYMBOL, timeframe="1m",
                                   context=self.live_context())
        self.assertEqual(result["status"], "symbol_mismatch")
        self.assertTrue(result["entry_blocked"])

    def test_missing_invalid_stale_and_future_data_are_distinct_blocks(self):
        missing = analyze_timeframe(None, symbol=SYMBOL, timeframe="1m",
                                   context=self.live_context())
        self.assertEqual(missing["data_status"], "invalid")
        malformed = candles()
        malformed = malformed.drop(columns=["volume"])
        malformed.attrs["symbol"] = SYMBOL
        result = analyze_timeframe(malformed, symbol=SYMBOL, timeframe="1m",
                                   context=self.live_context())
        self.assertEqual(result["status"], "invalid")
        stale = analyze_timeframe(
            candles(timestamp=pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=10)),
            symbol=SYMBOL, timeframe="1m", context=self.live_context())
        self.assertEqual(stale["status"], "stale")
        future = analyze_timeframe(
            candles(timestamp=pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=1)),
            symbol=SYMBOL, timeframe="1m", context=self.live_context())
        self.assertEqual(future["status"], "invalid")
        for blocked in (missing, result, stale, future):
            self.assertTrue(blocked["entry_blocked"])

    def test_valid_neutral_is_not_a_data_failure(self):
        result = analyze_timeframe(candles(), symbol=SYMBOL, timeframe="1m",
                                   context=self.live_context())
        self.assertEqual(result["status"], "neutral")
        self.assertEqual(result["data_status"], "valid")
        self.assertFalse(result["entry_blocked"])
        self.assertFalse(result["risk_veto"])

    def test_extreme_volatility_veto_survives_bullish_aggregation(self):
        veto = analyze_timeframe(candles(extreme=True), symbol=SYMBOL,
                                  timeframe="1m", context=self.live_context())
        self.assertEqual(veto["status"], "veto")
        self.assertTrue(veto["risk_veto"])
        bullish = {"signal": 1, "symbol": SYMBOL, "timeframe": "3m",
                   "status": "ok", "data_status": "valid",
                   "volatility_status": "expanding", "entry_blocked": False,
                   "risk_veto": False}
        aggregate = aggregate_results({"1m": veto, "3m": bullish}, symbol=SYMBOL,
                                      required_timeframes=("1m", "3m"))
        self.assertTrue(aggregate["entry_blocked"])
        self.assertTrue(aggregate["risk_veto"])
        self.assertEqual(aggregate["veto_timeframes"], ["1m"])
        self.assertEqual(aggregate["status"], "veto")
        self.assertEqual(aggregate["timeframe_results"]["3m"]["signal"], 1)

    def test_analyzer_exception_is_error_and_fail_closed(self):
        class ExplodingFrame(pd.DataFrame):
            @property
            def _constructor(self):
                return ExplodingFrame

            @property
            def loc(self):
                raise RuntimeError("fixture failure")

        frame = ExplodingFrame(candles())
        with patch("part7_signal._validate_frame", return_value=None), \
                patch("quantitative_math.part_signal", side_effect=RuntimeError("fixture failure")):
            result = analyze_timeframe(frame, symbol=SYMBOL, timeframe="1m",
                                       context=self.live_context())
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["data_status"], "error")
        self.assertTrue(result["entry_blocked"])

    def test_dashboard_exposes_part7_authoritative_fields(self):
        stream = io.StringIO()
        render_dashboard({"symbol": SYMBOL,
                          "signal": {"direction": "NO_TRADE", "confidence": 0},
                          "status": {"part7": {
                              "signal_name": "NEUTRAL",
                              "signal_identity": "part7:ETHUSDT:aggregate",
                              "reason": "Part7 data blocked on 1m",
                              "volatility_status": "unknown",
                              "data_status": "blocked",
                              "computation_backend": "pandas_cpu",
                              "entry_blocked": True,
                          }}, "reasons": ["Part7 data blocked on 1m"]},
                         stream=stream)
        output = stream.getvalue()
        for text in ("PART 7 SIGNAL", "part7:ETHUSDT:aggregate",
                     "PART 7 REASON", "PART 7 RISK", "PART 7 DATA", "pandas_cpu"):
            self.assertIn(text, output)


if __name__ == "__main__":
    unittest.main()
