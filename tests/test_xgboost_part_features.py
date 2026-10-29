"""Offline contract tests for the XGBoost 48-feature report and bridge."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xgboost as xgb

from jarvis_xgboost_engine import FEATURE_NAMES, JarvisXGBoostEngine


class _PredictorStub:
    def __init__(self, probability=0.9, feature_names=None, feature_count=48, raises=None):
        self.probability = probability
        self.feature_names = list(FEATURE_NAMES) if feature_names is None else feature_names
        self.feature_count = feature_count
        self.raises = raises
        self.called = False
        self.input_num_features = None
        self.input_feature_names = None

    def num_features(self):
        return self.feature_count

    def predict(self, dmatrix):
        self.called = True
        self.input_num_features = dmatrix.num_col()
        self.input_feature_names = dmatrix.feature_names
        if self.raises:
            raise self.raises
        return np.asarray([self.probability], dtype=np.float32)


class XGBoostFeatureReportTests(unittest.TestCase):
    def make_engine(self, path=None):
        return JarvisXGBoostEngine(model_path=path or "/no/such/jarvis-xgb-model.json", allow_bootstrap=False)

    def test_legacy_schema_and_missing_fields_are_explicit(self):
        engine = self.make_engine()
        report = engine.extract_feature_report({}, {}, {})

        self.assertEqual(len(FEATURE_NAMES), 48)
        self.assertEqual(report["feature_names"], FEATURE_NAMES)
        self.assertEqual(len(report["features"]), 48)
        self.assertEqual(len(report["feature_sources"]), 48)
        self.assertEqual(len(report["missing_features"]), 48)
        self.assertTrue(all(value == 0.0 for value in report["features"]))
        self.assertTrue(all(source is None for source in report["feature_sources"]))
        legacy = engine.extract_48_features({}, {}, {})
        self.assertEqual(legacy.shape, (48,))
        self.assertTrue(np.all(legacy == 0.0))

    def test_signal_fallbacks_and_part12_confidence_are_not_accuracy(self):
        engine = self.make_engine()
        parts = {
            "part4_volume": {"signal": 0.4, "telemetry": {"flow_label": "OHLCV proxy"}},
            "part12_confidence": {"confidence": 0.97, "telemetry": {"consensus_correctness_score_uncalibrated": 0.91}},
        }
        report = engine.extract_feature_report({}, parts, {})

        self.assertEqual(report["features"][9], 0.4)  # Existing P4 signal fallback only.
        self.assertIn("part4_volume.signal", report["feature_sources"][9])
        self.assertIn("p12_historical_accuracy_weight", report["missing_features"])
        self.assertEqual(report["features"][32], 0.0)
        self.assertIsNone(report["feature_sources"][32])
        self.assertIn("part12_confidence", report["parts_provided"])
        self.assertNotIn("part12_confidence", report["parts_used"])
        self.assertEqual(report["parts_participation"]["part12_confidence"], "present")

    def test_consensus_aliases_are_clipped_and_traced(self):
        engine = self.make_engine()
        report = engine.extract_feature_report(
            {},
            {"part11_fusion": {"consensus_ratio": 1.25, "vote_dispersion": 12.0}},
            {},
        )

        self.assertEqual(report["features"][30], 1.25)
        self.assertEqual(report["features"][31], 10.0)
        self.assertEqual(report["feature_sources"][30], "part11_fusion.consensus_ratio")
        self.assertEqual(report["feature_sources"][31], "part11_fusion.vote_dispersion")
        self.assertIn("part11_fusion", report["parts_used"])

    def test_part7_native_normalized_atr_is_used_in_matching_percent_units(self):
        engine = self.make_engine()
        report = engine.extract_feature_report(
            {},
            {"part7_volatility": {"signal": 0, "telemetry": {"norm_atr_pct": 0.83}}},
            {},
        )

        self.assertEqual(report["features"][18], 0.83)
        self.assertEqual(report["feature_sources"][18], "part7_volatility.telemetry.norm_atr_pct")
        self.assertNotIn("p7_atr_pct", report["missing_features"])
        self.assertIn("part7_volatility", report["parts_used"])

    def test_only_legacy_telemetry_fallbacks_are_used(self):
        engine = self.make_engine()
        report = engine.extract_feature_report(
            {"auxiliary_telemetry": {
                "funding_rate": {"funding_rate_zscore": -1.4},
                "order_book": {"imbalance_pct": 25.0},
            }},
            {},
            {},
        )

        self.assertEqual(report["features"][6], -1.4)
        self.assertEqual(report["features"][11], 0.25)
        self.assertIn("auxiliary_telemetry.funding_rate", report["feature_sources"][6])
        self.assertIn("imbalance_pct / 100", report["feature_sources"][11])
        # The similarly named macro/session slots have no synthesized neutral value.
        self.assertEqual(report["features"][47], 0.0)
        self.assertIn("macro_session_liquidity_state", report["missing_features"])

    def complete_inputs(self):
        parts = {
            "part1_breakout": {"breakout_intensity": 0.5, "squeeze_ratio": 0.8, "wick_ratio": 0.1},
            "part2_zone": {"demand_distance_pct": 1.2, "supply_distance_pct": 1.5, "retest_count": 2},
            "part3_psychology": {"funding_rate_zscore": -0.2, "retail_long_skew": 0.55, "liquidation_gravity": 0.2},
            "part4_volume": {"cvd_slope": 0.3, "relative_volume": 1.4, "bid_ask_imbalance": 0.1},
            "part5_ml": {"drift_zscore": 0.2, "trend_tstat": 1.1, "kalman_error": -0.1},
            "part6_trend": {"ema_alignment": 1.0, "adx_15m": 22, "adx_1h": 27},
            "part7_volatility": {"norm_atr_pct": 0.7, "parkinson_vol": 0.5, "choppiness_index": 40},
            "part8_structure": {"bos_choch": 1, "fvg_distance_pct": 0.2, "order_block_status": 1},
            "part9_orderflow": {"footprint_imbalance": 0.2, "iceberg_flag": 0, "delta_depth": 0.1},
            "part10_candlestats": {"wick_asymmetry": 0.3, "body_ratio": 0.5, "consecutive_bars": 2},
            "part11_fusion": {"quorum_ratio": 0.7, "dispersion_variance": 0.1},
            "part12_confidence": {"historical_accuracy_weight": 0.8},
        }
        options = {"telemetry": {
            "max_pain_distance_pct": 0.1, "max_pain_velocity_24h": 0.2,
            "pcr": 0.8, "pcr_oi": 0.9, "pcr_velocity_7d": -0.1,
            "net_dealer_gex": 2.0, "call_wall_distance_pct": 1.0,
            "put_wall_distance_pct": -1.0, "atm_iv": 50, "iv_rv_spread": 3,
            "expected_move_24h_pct": 2.5, "gamma_flip_distance_pct": 0.4,
        }}
        snapshot = {"eth_btc_momentum": 0.1, "btc_dominance_trend": -0.1, "session_liquidity": 0.8}
        return snapshot, parts, options

    def test_incomplete_features_veto_inference_even_when_model_exists(self):
        engine = self.make_engine()
        stub = _PredictorStub()
        engine.model = stub
        report = engine.evaluate_parts({}, {"part12_confidence": {"confidence": 0.8}}, {}, "BUY")

        self.assertFalse(report["available"])
        self.assertFalse(report["approved"])
        self.assertEqual(report["reason"], "missing_required_features")
        self.assertIsNone(report["inference"])
        self.assertIn("p12_historical_accuracy_weight", report["feature_report"]["missing_features"])
        self.assertFalse(stub.called)

    def test_complete_real_fields_reach_inference_and_are_reported(self):
        engine = self.make_engine()
        stub = _PredictorStub()
        engine.model = stub
        engine.model_source = "unit_test_stub"
        engine.training_status = "test_model_unvalidated"
        snapshot, parts, options = self.complete_inputs()
        report = engine.evaluate_parts(snapshot, parts, options, "BUY")

        self.assertTrue(report["available"])
        self.assertTrue(report["approved"])
        self.assertEqual(report["reason"], "evaluated")
        self.assertEqual(report["features_evaluated"], 48)
        self.assertEqual(report["feature_report"]["missing_features"], [])
        self.assertTrue(stub.called)
        self.assertEqual(stub.input_num_features, 48)
        self.assertEqual(stub.input_feature_names, FEATURE_NAMES)
        self.assertEqual(report["feature_report"]["parts_used"], list(parts.keys()))
        self.assertEqual(report["model_metadata"]["model_source"], "unit_test_stub")
        self.assertEqual(report["model_metadata"]["calibration_status"], "not_established")

    def test_real_xgboost_fixture_artifact_loads_and_predicts(self):
        # A tiny temporary test fixture, NOT a trading model or calibration.
        snapshot, parts, options = self.complete_inputs()
        extractor = self.make_engine()
        base = extractor.extract_48_features(snapshot, parts, options)
        rows = np.tile(base, (16, 1))
        rows[:, 0] = np.linspace(-1, 1, 16)
        labels = (rows[:, 0] > 0).astype(np.int32)
        matrix = xgb.DMatrix(rows, label=labels, feature_names=FEATURE_NAMES)
        fixture = xgb.train({'objective': 'binary:logistic', 'max_depth': 1,
                             'nthread': 1, 'verbosity': 0}, matrix, num_boost_round=3)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'test_fixture.json')
            fixture.save_model(path)
            engine = self.make_engine(path)
            result = engine.evaluate_parts(snapshot, parts, options, 'BUY')
            self.assertTrue(result['available'])
            self.assertEqual(result['features_evaluated'], 48)
            self.assertEqual(result['model_metadata']['model_source'], 'local_model_artifact')
            self.assertEqual(result['model_metadata']['calibration_status'], 'not_established')
            self.assertGreaterEqual(result['win_probability'], 0)
            self.assertLessEqual(result['win_probability'], 1)

    def test_bootstrap_disabled_never_creates_model_or_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "missing.json")
            engine = self.make_engine(path)
            result = engine.evaluate(np.zeros(48, dtype=np.float32))

            self.assertIsNone(engine.model)
            self.assertFalse(engine.is_bootstrapped)
            self.assertEqual(engine.training_status, "unavailable")
            self.assertFalse(Path(path).exists())
            self.assertFalse(result["available"])
            self.assertFalse(result["approved"])
            self.assertEqual(result["reason"], "model_unavailable")

    def test_evaluate_rejects_non_finite_features_and_invalid_probabilities(self):
        engine = self.make_engine()
        stub = _PredictorStub()
        engine.model = stub
        bad_features = engine.evaluate(np.full(48, np.nan, dtype=np.float32))
        self.assertFalse(bad_features["available"])
        self.assertFalse(bad_features["approved"])
        self.assertEqual(bad_features["reason"], "non_finite_features")
        self.assertFalse(stub.called)

        for probability in (np.nan, np.inf, -0.01, 1.01):
            engine = self.make_engine()
            stub = _PredictorStub(probability=probability)
            engine.model = stub
            result = engine.evaluate(np.zeros(48, dtype=np.float32))
            self.assertFalse(result["available"], probability)
            self.assertFalse(result["approved"], probability)
            self.assertEqual(result["reason"], "invalid_prediction_probability")
            self.assertTrue(stub.called)

    def test_directional_score_never_approves_sell_on_bullish_prediction(self):
        snapshot, parts, options = self.complete_inputs()
        cases = [('BUY', .9, True), ('BUY', .1, False),
                 ('SELL', .9, False), ('SELL', .1, True),
                 ('SHORT', .9, False), ('PUT', .1, True)]
        for direction, raw, expected in cases:
            with self.subTest(direction=direction, raw=raw):
                engine = self.make_engine(); engine.model = _PredictorStub(probability=raw)
                engine.model_source = 'unit_test_stub'
                report = engine.evaluate_parts(snapshot, parts, options, direction)
                self.assertTrue(report['available'])
                self.assertEqual(report['approved'], expected)
                score = 1 - raw if direction in ('SELL', 'SHORT', 'PUT') else raw
                self.assertAlmostEqual(report['win_probability'], score, places=4)
        engine = self.make_engine(); engine.model = _PredictorStub()
        report = engine.evaluate_parts(snapshot, parts, options, 'UNKNOWN')
        self.assertFalse(report['available'])
        self.assertEqual(report['reason'], 'invalid_trade_direction')

    def test_evaluate_requires_a_single_prediction_value(self):
        engine = self.make_engine()
        stub = _PredictorStub(probability=[0.7, 0.8])
        engine.model = stub

        result = engine.evaluate(np.zeros(48, dtype=np.float32))

        self.assertFalse(result["available"])
        self.assertFalse(result["approved"])
        self.assertEqual(result["reason"], "invalid_prediction_count")
        self.assertTrue(stub.called)

    def test_model_feature_count_and_names_must_match_artifact_schema(self):
        cases = [
            (_PredictorStub(feature_count=47), "model_feature_count_mismatch"),
            (_PredictorStub(feature_names=list(FEATURE_NAMES[:-1]) + ["wrong_feature"]), "model_feature_names_mismatch"),
        ]
        missing_names = _PredictorStub()
        missing_names.feature_names = None
        cases.append((missing_names, "model_feature_names_unavailable"))

        for stub, reason in cases:
            engine = self.make_engine()
            engine.model = stub
            result = engine.evaluate(np.zeros(48, dtype=np.float32))
            self.assertFalse(result["available"])
            self.assertFalse(result["approved"])
            self.assertEqual(result["reason"], reason)
            self.assertFalse(stub.called)

    def test_evaluate_parts_reports_predict_exceptions_as_unavailable(self):
        engine = self.make_engine()
        stub = _PredictorStub(raises=RuntimeError("predict failed"))
        engine.model = stub
        engine.model_source = "local_model_artifact"
        engine.training_status = "artifact_loaded_unvalidated"
        snapshot, parts, options = self.complete_inputs()

        result = engine.evaluate_parts(snapshot, parts, options, "BUY")

        self.assertFalse(result["available"])
        self.assertFalse(result["approved"])
        self.assertEqual(result["reason"], "inference_exception")
        self.assertIn("predict failed", result["inference_error"])
        self.assertTrue(stub.called)

    def test_synthetic_bootstrap_provenance_is_uncalibrated_and_not_persisted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "synthetic.json")
            random_state = np.random.get_state()
            try:
                with patch("jarvis_xgboost_engine.xgb.train", return_value=_PredictorStub()):
                    engine = JarvisXGBoostEngine(model_path=path, allow_bootstrap=True)
            finally:
                np.random.set_state(random_state)

            self.assertIsNotNone(engine.model)
            self.assertTrue(engine.is_bootstrapped)
            self.assertEqual(engine.model_source, "synthetic_priors")
            self.assertEqual(engine.training_status, "synthetic_prior_fit_uncalibrated")
            self.assertFalse(engine.approved_for_advisory)
            self.assertFalse(Path(path).exists())
            snapshot, parts, options = self.complete_inputs()
            result = engine.evaluate_parts(snapshot, parts, options, "BUY")
            self.assertFalse(result["available"])
            self.assertFalse(result["approved"])
            self.assertEqual(result["reason"], "synthetic_model_not_advisory")
            self.assertFalse(engine.model.called)


if __name__ == "__main__":
    unittest.main()
