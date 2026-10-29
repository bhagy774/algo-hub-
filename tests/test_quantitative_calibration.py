"""Part 7 veto must scale with the frame's own volatility, not a fixed 1.5%."""
import math
import quantitative_math as q


def _rows(n, half_range_pct, drift=0.0):
    rows, price = [], 100.0
    for i in range(n):
        o = price
        price = price * (1 + drift + 0.0005 * math.sin(i))
        rows.append({"open": o, "high": max(o, price) * (1 + half_range_pct),
                     "low": min(o, price) * (1 - half_range_pct), "close": price, "volume": 1000.0})
    return rows


def test_steady_wide_frame_is_not_vetoed():
    # 4h-like frame: every candle spans ~2%; nothing is expanding, so no veto.
    sig = q.part_signal("7", _rows(200, 0.01))
    assert not sig["telemetry"].get("risk_veto")


def test_absolute_extreme_range_still_vetoes():
    sig = q.part_signal("7", _rows(200, 0.06))
    assert sig["telemetry"].get("risk_veto") is True
