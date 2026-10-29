import math
import unittest

import quantitative_math as qm
from neural_parts import PARTS, FEATURE_NAMES, prepare_features


def fixture(n=160, *, flat=False, zero_volume=False):
    rows=[]
    price=100.0
    for i in range(n):
        if flat:
            o=c=price
            span=0.0
        else:
            o=price
            price=o*(1+0.00035+0.0012*math.sin(i*0.37))
            c=price
            span=o*(0.0005+0.001*abs(math.sin(i*0.19)))
        rows.append({"open":o,"high":max(o,c)+span,"low":min(o,c)-span,
                     "close":c,"volume":0.0 if zero_volume else float(50+i%17)})
    return rows


class QuantitativeMathTests(unittest.TestCase):
    def test_all_ten_owned_feature_projection_contracts_are_finite_and_stable(self):
        rows=fixture()
        for part in PARTS[:10]:
            with self.subTest(part=part):
                x=prepare_features(part,rows)
                self.assertEqual(len(x),len(FEATURE_NAMES[part]))
                self.assertTrue(all(math.isfinite(v) for v in x))
                self.assertEqual(x,prepare_features(part,rows))
                result=qm.part_signal(str(PARTS.index(part)+1),rows)
                self.assertIn(result["signal"],(-1,0,1))
                self.assertTrue(math.isfinite(result["confidence"]))

    def test_invalid_or_insufficient_bars_fail_closed(self):
        self.assertFalse(qm.quantitative_features(fixture(1)).get("available"))
        bad=fixture()
        bad[-1]["close"]=float("nan")
        self.assertFalse(qm.quantitative_features(bad).get("available"))
        bad=fixture()
        bad[-1]["high"]=bad[-1]["low"]-1
        self.assertFalse(qm.quantitative_features(bad).get("available"))
        with self.assertRaises(ValueError):
            prepare_features("part1_breakout",fixture(30))

    def test_flat_price_and_zero_volume_edges_are_finite(self):
        flat=fixture(flat=True)
        f=qm.quantitative_features(flat)
        self.assertTrue(f["available"])
        for key in ("realized_vol","downside_vol","jump_share","trend_score","volume_surprise_robust_z","flow_imbalance_proxy"):
            self.assertTrue(math.isfinite(f[key]),key)
        z=fixture(zero_volume=True)
        self.assertFalse(qm.price_volume_distribution(z).get("available"))
        self.assertTrue(all(math.isfinite(x) for x in prepare_features("part4_volume",z)))

    def test_original_part1_market_data_envelope_is_supported(self):
        rows=fixture(80)
        wrapped={"price_action":rows,"volume_pattern":[r["volume"] for r in rows]}
        self.assertEqual(qm.quantitative_features(wrapped)["n"],80)
        self.assertEqual(qm.part_signal("1",wrapped),qm.part_signal("1",rows))

    def test_risk_scale_is_price_unit_positive_and_has_conservative_floor(self):
        rows=fixture()
        price=rows[-1]["close"]
        minimum=price*0.002
        self.assertGreaterEqual(qm.risk_scale_price(rows,price,floor_pct=0.002),minimum)
        self.assertGreaterEqual(qm.risk_scale_price(None,price,floor_pct=0.002),minimum)
        self.assertEqual(qm.risk_scale_price(rows,0),0.0)

    def test_close_series_trend_is_robust_and_fail_closed(self):
        rising=[100.0*math.exp(0.002*i) for i in range(80)]
        falling=list(reversed(rising))
        self.assertGreater(qm.close_return_trend(rising)["trend_score"],0)
        self.assertLess(qm.close_return_trend(falling)["trend_score"],0)
        self.assertFalse(qm.close_return_trend([100.0,float("nan"),101.0])["available"])

    def test_part7_live_snapshot_is_bounded_cpu_native_and_fail_closed(self):
        rows=fixture(180)
        snapshot=qm.live_feature_snapshot(rows)
        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["n_closed_bars"],128)
        self.assertEqual(snapshot["compute_device"],"cpu")
        self.assertEqual(snapshot["signal_role"],"advisory_only_no_execution_authority")
        self.assertEqual(len(snapshot["log_returns"]),96)
        self.assertIn("not trade-tape",snapshot["flow_proxy_label"])
        self.assertEqual(qm.live_feature_snapshot(fixture(1))["available"],False)
        self.assertEqual(qm.live_directional_advisory({"available":False}),"HOLD")
        directional={"available":True,"state_velocity_z":2.0,"robust_trend_score":1.5,
                     "change_point_bic_gain":3.0,"high_variance_state_posterior":0.2,
                     "short_long_variance_ratio":1.0}
        self.assertEqual(qm.live_directional_advisory(directional),"BUY")
        self.assertEqual(qm.live_directional_advisory(dict(directional,high_variance_state_posterior=.95)),"HOLD")
        self.assertEqual(qm.live_directional_advisory(dict(directional,short_long_variance_ratio=3.0)),"HOLD")

    def test_proxy_labels_are_explicit_and_no_random_scores_are_used(self):
        f=qm.quantitative_features(fixture())
        self.assertIn("not trade-tape",f["flow_label"])
        p=qm.price_volume_distribution(fixture())
        self.assertIn("not traded-at-price",p["profile_label"])
        a=qm.part_signal("9",fixture())
        self.assertFalse(a["telemetry"]["flow_is_measured_orderbook"])

    def test_part7_extreme_realized_variance_is_neutral_but_vetoes_entry(self):
        rows=[]
        price=100.0
        for i in range(100):
            ret=0.0002*math.sin(i*.8) if i < 90 else 0.02*(-1 if i % 2 else 1)
            open_price=price
            price=open_price*math.exp(ret)
            rows.append({"open":open_price,"high":max(open_price,price)*1.0002,
                         "low":min(open_price,price)*.9998,"close":price,"volume":100.0})
        result=qm.part_signal("7",rows)
        self.assertEqual(result["signal"],0)
        self.assertTrue(result["telemetry"]["risk_veto"])
        self.assertTrue(result["telemetry"]["entry_blocked"])
        self.assertGreaterEqual(result["telemetry"]["high_state_posterior"],0.90)
        self.assertTrue(math.isfinite(result["confidence"]))


if __name__ == "__main__":
    unittest.main()
