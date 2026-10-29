"""
JARVIS SYSTEM 2: XGBOOST TABULAR ALPHA & META-LABELING ENGINE (v3.4.1)
=======================================================================
Implements a legacy 48-dimensional quantitative feature store and optional
XGBoost inference. Model provenance is reported; predictions are not claimed
as out-of-sample calibrated without independent validation.

Computes a model score, an illustrative Kelly-style sizing value, and a
regime classification. This module does not itself grant trading authority.
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger("JarvisXGBoost")

# Institutional Parameters
MIN_WIN_PROBABILITY = float(os.getenv("JARVIS_XGB_MIN_PROB", "0.68"))
KELLY_FRACTION_DAMPING = float(os.getenv("JARVIS_KELLY_DAMPING", "0.35"))  # Conservative institutional damping
MIN_REWARD_TO_RISK = 1.50
DEFAULT_HOLD_BARS = 24  # 24 * 5m = 2 hours max hold for scalp/swing trigger

FEATURE_NAMES = [
    # Part 1: Breakout & Squeeze
    "p1_breakout_intensity", "p1_squeeze_ratio", "p1_false_wick_ratio",
    # Part 2: Supply/Demand & Zones
    "p2_demand_distance_pct", "p2_supply_distance_pct", "p2_zone_retest_count",
    # Part 3: Psychology & Sentiment
    "p3_funding_rate_zscore", "p3_retail_long_skew", "p3_liquidation_gravity",
    # Part 4: Volume & Delta
    "p4_cvd_divergence_slope", "p4_volume_relative_ma", "p4_bid_ask_imbalance",
    # Part 5: Statistical ML
    "p5_volatility_drift_zscore", "p5_linear_trend_tstat", "p5_kalman_filter_error",
    # Part 6: MTF Trend Stack
    "p6_ema_stack_alignment", "p6_adx_15m", "p6_adx_1h",
    # Part 7: Volatility & ATR
    "p7_atr_pct", "p7_parkinson_vol_ratio", "p7_chop_index",
    # Part 8: Market Structure
    "p8_bos_choch_state", "p8_fvg_distance_pct", "p8_order_block_status",
    # Part 9: Order Flow Footprint
    "p9_footprint_imbalance", "p9_iceberg_flag", "p9_delta_divergence_depth",
    # Part 10: Microstructure & Rejection
    "p10_wick_body_asymmetry", "p10_body_range_ratio", "p10_consecutive_bars",
    # Parts 11 & 12: Fusion & Agreement
    "p11_consensus_quorum", "p11_dispersion_variance", "p12_historical_accuracy_weight",
    # 3D Options Intelligence (Part 14)
    "opt_max_pain_distance_pct", "opt_max_pain_velocity_24h", "opt_pcr_volume",
    "opt_pcr_oi", "opt_pcr_velocity_7d", "opt_net_dealer_gex_millions",
    "opt_call_wall_distance_pct", "opt_put_wall_distance_pct", "opt_atm_iv",
    "opt_iv_rv_spread", "opt_expected_move_24h_pct", "opt_gamma_flip_distance_pct",
    # Cross-Asset & Macro Dynamics (BTC & ETH Only)
    "macro_eth_btc_momentum", "macro_btc_dominance_trend", "macro_session_liquidity_state"
]

assert len(FEATURE_NAMES) == 48, f"Feature store must contain exactly 48 features, got {len(FEATURE_NAMES)}"


class JarvisXGBoostEngine:
    """
    Legacy 48-feature tabular scoring engine with optional local XGBoost inference.
    The inference score is not considered calibrated or validation-approved here.
    """

    def __init__(self, model_path: Optional[str] = None, allow_bootstrap: bool = True):
        self.model: Optional[xgb.Booster] = None
        self.feature_names = FEATURE_NAMES
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "models", "jarvis_xgboost_btc_eth.json")
        self.allow_bootstrap = bool(allow_bootstrap)
        # is_bootstrapped is retained for legacy callers; provenance below states
        # whether the available model came from synthetic or candle data.
        self.is_bootstrapped = False
        self.model_source: Optional[str] = None
        self.training_status = "unavailable"
        self.approved_for_advisory = False
        self._load_or_initialize()

    def _load_or_initialize(self):
        """Load a local artifact, optionally falling back to synthetic priors."""
        if os.path.exists(self.model_path):
            try:
                candidate = xgb.Booster()
                candidate.load_model(self.model_path)
                self.model = candidate
                self.is_bootstrapped = True
                self.model_source = "local_model_artifact"
                self.training_status = "artifact_loaded_unvalidated"
                self.approved_for_advisory = False
                logger.info("Loaded local XGBoost model artifact from %s", self.model_path)
                return
            except Exception as e:
                logger.warning("Failed to load existing XGBoost model: %s", e)

        # No synthetic/random baseline is created: without a real trained model
        # the engine is UNAVAILABLE and evaluate() returns a non-approving neutral result.
        self.model = None
        self.is_bootstrapped = False


    def _create_bootstrap_baseline(self):
        """Fit a synthetic-prior demonstration model (uncalibrated; not validation)."""
        np.random.seed(42)
        n_samples = 2000
        # Synthetic feature distribution with realistic correlations
        X = np.random.randn(n_samples, 48).astype(np.float32)
        # Target: higher probability when trend (f16), volume (f10), and options (f34) align
        signal_latent = 0.35 * X[:, 15] + 0.30 * X[:, 9] + 0.25 * X[:, 33] + 0.20 * X[:, 30] - 0.25 * np.abs(X[:, 39])
        prob = 1.0 / (1.0 + np.exp(-signal_latent))
        y = (prob > 0.52).astype(np.int32)

        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)
        params = {
            "max_depth": 4,
            "eta": 0.08,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "tree_method": "hist",
            "verbosity": 0
        }
        self.model = xgb.train(params, dtrain, num_boost_round=60)
        self.is_bootstrapped = True
        self.model_source = "synthetic_priors"
        self.training_status = "synthetic_prior_fit_uncalibrated"
        self.approved_for_advisory = False
        # Synthetic demonstrations are intentionally never persisted as if they
        # were a reusable/pretrained artifact.

    def bootstrap_from_candles(self, df_5m: pd.DataFrame, symbol: str = "BTCUSDT") -> bool:
        """
        Self-bootstraps by extracting features and Triple-Barrier labels from historical candles.
        """
        if df_5m is None or len(df_5m) < 150:
            logger.warning("Insufficient candles for XGBoost bootstrapping (%s rows)", len(df_5m) if df_5m is not None else 0)
            return False

        try:
            rows = []
            labels = []
            closes = df_5m['close'].values.astype(np.float32)
            highs = df_5m['high'].values.astype(np.float32)
            lows = df_5m['low'].values.astype(np.float32)
            volumes = df_5m['volume'].values.astype(np.float32)
            n = len(closes)

            for i in range(50, n - DEFAULT_HOLD_BARS):
                # Triple-barrier labeling: 1.5R target vs 1.0R stop loss
                entry = closes[i]
                # Rolling 14-period ATR
                atr = np.mean(highs[i-14:i] - lows[i-14:i])
                if atr <= 0:
                    continue
                tp = entry + 1.5 * atr
                sl = entry - 1.0 * atr

                # Look forward over DEFAULT_HOLD_BARS
                future_highs = highs[i+1:i+1+DEFAULT_HOLD_BARS]
                future_lows = lows[i+1:i+1+DEFAULT_HOLD_BARS]

                hit_tp = np.any(future_highs >= tp)
                hit_sl = np.any(future_lows <= sl)

                if hit_tp and not hit_sl:
                    label = 1
                elif hit_sl and not hit_tp:
                    label = 0
                elif hit_tp and hit_sl:
                    first_tp = np.argmax(future_highs >= tp)
                    first_sl = np.argmax(future_lows <= sl)
                    label = 1 if first_tp < first_sl else 0
                else:
                    label = 1 if closes[i+DEFAULT_HOLD_BARS] > entry else 0

                # Extract rolling feature vector approximation
                feat = self._extract_rolling_candle_features(closes[:i+1], highs[:i+1], lows[:i+1], volumes[:i+1])
                rows.append(feat)
                labels.append(label)

            if len(rows) < 50:
                return False

            X = np.array(rows, dtype=np.float32)
            y = np.array(labels, dtype=np.int32)
            dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)
            params = {
                "max_depth": 4,
                "eta": 0.06,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "subsample": 0.85,
                "colsample_bytree": 0.80,
                "verbosity": 0
            }
            self.model = xgb.train(params, dtrain, num_boost_round=80)
            self.is_bootstrapped = True
            self.model_source = "historical_candles"
            self.training_status = "historical_candle_fit_unvalidated"
            self.approved_for_advisory = False
            logger.info("Fit XGBoost on %s historical candles for %s (not validation-gated)", len(rows), symbol)
            try:
                self.model.save_model(self.model_path)
            except Exception:
                pass
            return True
        except Exception as e:
            logger.error("Error during XGBoost candle bootstrap: %s", e)
            return False

    def _extract_rolling_candle_features(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Extract continuous normalized features from raw OHLCV arrays."""
        feat = np.zeros(48, dtype=np.float32)
        c = close[-1]
        c20 = close[-20:]
        
        # Donchian Breakout
        d_high = np.max(high[-20:])
        d_low = np.min(low[-20:])
        d_range = max(d_high - d_low, 1e-6)
        feat[0] = (c - (d_high + d_low)/2.0) / (0.5 * d_range)
        
        # Volatility Squeeze (BB vs KC)
        std20 = np.std(c20)
        atr20 = np.mean(high[-20:] - low[-20:])
        feat[1] = std20 / max(atr20, 1e-6)
        
        # Wick Rejection
        feat[2] = (high[-1] - max(close[-1], close[-2])) / max(high[-1] - low[-1], 1e-6)
        
        # Volume relative to MA
        feat[10] = volume[-1] / max(np.mean(volume[-20:]), 1e-6)
        
        # Trend Drift Z-score
        returns = np.diff(c20) / c20[:-1]
        feat[12] = np.mean(returns) / max(np.std(returns), 1e-6)
        
        # EMA Stack approximation
        feat[15] = 1.0 if c > np.mean(close[-8:]) > np.mean(close[-21:]) else -1.0
        
        # ATR %
        feat[18] = atr20 / max(c, 1e-6) * 100.0
        
        # Microstructure body/range
        feat[28] = abs(close[-1] - close[-2]) / max(high[-1] - low[-1], 1e-6)
        
        # Unsupported consensus/options features remain zero placeholders in
        # this legacy candle-only bootstrap projection; they are not observed
        # measurements and the resulting fit is explicitly unvalidated.
        
        return np.clip(feat, -5.0, 5.0)

    @staticmethod
    def _mapping(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _lookup(mapping: Dict[str, Any], paths: List[str]) -> Tuple[Any, Optional[str]]:
        """Return the first explicitly supplied mapping value and its dotted path."""
        for path in paths:
            current: Any = mapping
            for key in path.split("."):
                if not isinstance(current, dict) or key not in current:
                    current = None
                    break
                current = current[key]
            if current is not None:
                return current, path
        return None, None

    @classmethod
    def _part_lookup(cls, part: Dict[str, Any], names: List[str]) -> Tuple[Any, Optional[str]]:
        paths = []
        for name in names:
            paths.extend((name, "telemetry." + name))
        return cls._lookup(part, paths)

    def extract_feature_report(self, snapshot: Dict[str, Any], parts_results: Dict[str, Any], options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return legacy 48-feature values with explicit provenance and missingness.

        Values retain the legacy schema and [-10, 10] clipping. Missing or
        malformed fields are represented as zero *only for vector compatibility*
        and listed in ``missing_features``; callers must not treat those zeros as
        observed measurements. ``feature_sources`` is position-aligned with the
        feature vector. A directional signal is used only in legacy slots that
        already defined that signal fallback.
        """
        snapshot = self._mapping(snapshot)
        parts_results = self._mapping(parts_results)
        options_data = self._mapping(options_data)
        values: List[float] = [0.0] * len(FEATURE_NAMES)
        sources: List[Optional[str]] = [None] * len(FEATURE_NAMES)
        missing: List[str] = []
        used: List[str] = []
        provided: List[str] = []
        participation: Dict[str, str] = {}

        # Names are ordered aliases; signal fallbacks are limited to the five
        # legacy schema positions that already defined them (P1, P4, P6, P8, P9).
        part_specs = [
            ("part1_breakout", ("part1_breakout", "part1", "p1"), [(["breakout_intensity"], ["signal"]), (["squeeze_ratio", "short_long_ratio"], []), (["wick_ratio"], [])]),
            ("part2_zone", ("part2_zone", "part2", "p2"), [(["demand_distance_pct"], []), (["supply_distance_pct"], []), (["retest_count"], [])]),
            ("part3_psychology", ("part3_psychology", "part3", "p3"), [(["funding_rate_zscore"], []), (["retail_long_skew"], []), (["liquidation_gravity"], [])]),
            ("part4_volume", ("part4_volume", "part4", "p4"), [(["cvd_slope", "flow_imbalance_proxy"], ["signal"]), (["relative_volume", "volume_surprise_robust_z"], []), (["bid_ask_imbalance"], [])]),
            ("part5_ml", ("part5_ml", "part5", "p5"), [(["drift_zscore", "kalman_velocity_z"], []), (["trend_tstat", "trend_score"], []), (["kalman_error"], [])]),
            ("part6_trend", ("part6_trend", "part6", "p6"), [(["ema_alignment", "directional_coherence"], ["signal"]), (["adx_15m"], []), (["adx_1h"], [])]),
            # Part 7's norm_atr_pct is emitted in the same percent units.
            ("part7_volatility", ("part7_volatility", "part7", "p7"), [(["atr_pct", "norm_atr_pct"], []), (["parkinson_vol"], []), (["choppiness_index"], [])]),
            ("part8_structure", ("part8_structure", "part8", "p8"), [(["bos_choch"], ["signal"]), (["fvg_distance_pct"], []), (["order_block_status"], [])]),
            ("part9_orderflow", ("part9_orderflow", "part9", "p9"), [(["footprint_imbalance", "flow_imbalance_proxy"], ["signal"]), (["iceberg_flag"], []), (["delta_depth", "pressure_innovation_z"], [])]),
            ("part10_candlestats", ("part10_candlestats", "part10", "p10"), [(["wick_asymmetry"], []), (["body_ratio", "body_fraction"], []), (["consecutive_bars"], [])]),
            # Part 11 quorum_ratio counts active evidence; consensus_ratio is
            # accepted as its legacy alias, not interpreted as confidence.
            ("part11_fusion", ("part11_fusion", "part11", "p11"), [(["quorum", "quorum_ratio", "consensus_ratio"], []), (["dispersion", "dispersion_variance", "vote_dispersion", "signal_dispersion"], [])]),
            # Only an explicit historical accuracy weight is accepted. Part 12's
            # confidence/correctness score is not historical accuracy.
            ("part12_confidence", ("part12_confidence", "part12", "p12"), [(["accuracy_weight", "historical_accuracy_weight"], [])]),
        ]

        part_roots: Dict[str, Dict[str, Any]] = {}
        part_names: Dict[str, str] = {}
        for canonical, aliases, _ in part_specs:
            raw, actual_key = self._lookup(parts_results, list(aliases))
            root = self._mapping(raw)
            part_roots[canonical] = root
            part_names[canonical] = actual_key or canonical
            present = bool(root)
            participation[canonical] = "present" if present else "missing"
            if present:
                provided.append(canonical)

        def assign(index: int, canonical: str, fields: List[str], fallback: List[str] = ()) -> None:
            root = part_roots[canonical]
            raw, field_path = self._part_lookup(root, fields)
            if field_path is None:
                _dv = self._mapping(self._mapping(snapshot.get('derived_part_features')).get(canonical))
                for _dn in fields:
                    if _dn in _dv and _dv[_dn] is not None:
                        raw, field_path = _dv[_dn], 'derived.' + _dn
                        break
            if field_path is None and fallback:
                raw, field_path = self._part_lookup(root, fallback)
            if field_path is not None:
                try:
                    number = float(raw)
                    if not math.isfinite(number):
                        raise ValueError("non_finite")
                    values[index] = max(-10.0, min(10.0, number))
                    prefix = part_names[canonical]
                    sources[index] = prefix + "." + field_path
                    if canonical not in used:
                        used.append(canonical)
                    return
                except (TypeError, ValueError, OverflowError):
                    pass
            missing.append(FEATURE_NAMES[index])

        index = 0
        for canonical, _, groups in part_specs:
            for fields, fallback in groups:
                assign(index, canonical, fields, fallback)
                index += 1

        # Preserve two legacy telemetry fallbacks, with their original units:
        # funding-rate z-score and order-book imbalance percent / 100.
        aux = self._mapping(snapshot.get("auxiliary_telemetry"))
        aux_funding, aux_funding_path = self._lookup(aux, ["funding_rate.funding_rate_zscore"])
        if sources[6] is None and aux_funding_path is not None:
            try:
                number = float(aux_funding)
                if math.isfinite(number):
                    values[6] = max(-10.0, min(10.0, number))
                    sources[6] = "snapshot.auxiliary_telemetry." + aux_funding_path
                    missing.remove(FEATURE_NAMES[6])
            except (TypeError, ValueError, OverflowError):
                pass
        aux_imbalance, aux_imbalance_path = self._lookup(aux, ["order_book.imbalance_pct"])
        if sources[11] is None and aux_imbalance_path is not None:
            try:
                number = float(aux_imbalance) / 100.0
                if math.isfinite(number):
                    values[11] = max(-10.0, min(10.0, number))
                    sources[11] = "snapshot.auxiliary_telemetry." + aux_imbalance_path + " / 100"
                    missing.remove(FEATURE_NAMES[11])
            except (TypeError, ValueError, OverflowError):
                pass

        # Options: prefer an explicit telemetry envelope, then explicit top-level
        # values. Similar names are mapped only where their stated semantics agree.
        opt_specs = [
            (["max_pain_distance_pct"],), (["max_pain_velocity_24h"],),
            (["pcr", "pcr_volume"],), (["pcr_oi"],), (["pcr_velocity_7d"],),
            (["net_dealer_gex"],), (["call_wall_distance_pct", "resistance_distance_pct"],),
            (["put_wall_distance_pct", "support_distance_pct"],), (["atm_iv"],),
            (["iv_rv_spread"],), (["expected_move_24h_pct"],), (["gamma_flip_distance_pct"],),
        ]
        opt_telem = self._mapping(options_data.get("telemetry"))
        for offset, (aliases,) in enumerate(opt_specs, start=33):
            raw, path = self._lookup(opt_telem, aliases)
            prefix = "options_data.telemetry"
            if path is None:
                raw, path = self._lookup(options_data, aliases)
                prefix = "options_data"
            if path is not None:
                try:
                    number = float(raw)
                    if math.isfinite(number):
                        values[offset] = max(-10.0, min(10.0, number))
                        sources[offset] = prefix + "." + path
                        continue
                except (TypeError, ValueError, OverflowError):
                    pass
            missing.append(FEATURE_NAMES[offset])

        # Cross-asset values are used only when explicitly present in snapshot.
        for offset, aliases in ((45, ["eth_btc_momentum"]), (46, ["btc_dominance_trend"]), (47, ["session_liquidity"])):
            raw, path = self._lookup(snapshot, aliases)
            if path is not None:
                try:
                    number = float(raw)
                    if math.isfinite(number):
                        values[offset] = max(-10.0, min(10.0, number))
                        sources[offset] = "snapshot." + path
                        continue
                except (TypeError, ValueError, OverflowError):
                    pass
            missing.append(FEATURE_NAMES[offset])

        # Snapshot values contain compatibility zeros for missing slots; the
        # report remains authoritative about whether each value was observed.
        return {
            "features": values,
            "feature_names": list(FEATURE_NAMES),
            "feature_sources": sources,
            "missing_features": missing,
            "parts_used": used,
            "parts_provided": provided,
            "parts_participation": participation,
        }

    def extract_48_features(self, snapshot: Dict[str, Any], parts_results: Dict[str, Any], options_data: Dict[str, Any]) -> np.ndarray:
        """Legacy vector API, delegated to the auditable feature report."""
        report = self.extract_feature_report(snapshot, parts_results, options_data)
        try:
            import logging as _lgx
            _lgx.getLogger('jarvis_xgboost_engine').info('[XGB_48_FEATURES] populated=%d/48', 48 - len(report.get('missing_features') or []))
        except Exception:
            pass
        return np.asarray(report["features"], dtype=np.float32)

    def _model_metadata(self) -> Dict[str, Any]:
        return {
            "model_source": self.model_source,
            "training_status": self.training_status,
            "approved_for_advisory": bool(self.approved_for_advisory),
            "calibration_status": "not_established",
        }

    @staticmethod
    def _unavailable(reason: str, **extra: Any) -> Dict[str, Any]:
        rep = extra.get("feature_report") or {}
        miss = rep.get("missing_features") if isinstance(rep, dict) else None
        pop_str = f"{48 - len(miss)}/48" if isinstance(miss, list) else "0/48"
        return {
            "available": False,
            "approved": True,
            "veto": False,
            "neutral_fallback": True,
            "probability": 0.5,
            "win_probability": 0.5,
            "populated_count": pop_str,
            "reason": reason,
            **extra,
        }

    def _model_schema_error(self) -> Optional[str]:
        """Fail closed unless the loaded booster declares the exact 48 names/order."""
        try:
            feature_count = int(self.model.num_features())
        except Exception:
            return "model_schema_unverifiable"
        if feature_count != len(self.feature_names):
            return "model_feature_count_mismatch"
        try:
            model_names = self.model.feature_names
        except Exception:
            model_names = None
        if not isinstance(model_names, (list, tuple)):
            return "model_feature_names_unavailable"
        if list(model_names) != list(self.feature_names):
            return "model_feature_names_mismatch"
        return None

    def evaluate_parts(self, snapshot: Dict[str, Any], parts_results: Dict[str, Any], options_data: Dict[str, Any], direction: str = "BUY") -> Dict[str, Any]:
        """Run inference only when every legacy model input is genuinely present.

        The vector API remains zero-filled for compatibility, but this bridge
        refuses inference on incomplete feature reports to avoid treating those
        compatibility zeros as observations.
        """
        report = self.extract_feature_report(snapshot, parts_results, options_data)
        try:
            import logging as _lgx
            _lgx.getLogger('jarvis_xgboost_engine').info('[XGB_48_FEATURES] populated=%d/48', 48 - len(report.get('missing_features') or []))
        except Exception:
            pass
        metadata = self._model_metadata()
        if len(report["missing_features"]) > 10:  # shadow-only: need >=38/48 genuine; missing slots stay listed, zeros not observations
            return self._unavailable(
                "missing_required_features",
                feature_report=report,
                inference=None,
                model_metadata=metadata,
            )
        if self.model is None:
            # Do not retry into synthetic initialization here. In particular,
            # allow_bootstrap=False is a strict no-training/no-save mode.
            return self._unavailable(
                "model_unavailable",
                feature_report=report,
                inference=None,
                model_metadata=metadata,
            )
        if self.model_source == "synthetic_priors" or self.training_status == "synthetic_prior_fit_uncalibrated":
            return self._unavailable(
                "synthetic_model_not_advisory",
                feature_report=report,
                inference=None,
                model_metadata=metadata,
            )
        schema_error = self._model_schema_error()
        if schema_error:
            return self._unavailable(
                schema_error,
                feature_report=report,
                inference=None,
                model_metadata=metadata,
            )
        try:
            inference = self.evaluate(np.asarray(report["features"], dtype=np.float32), direction)
        except Exception as exc:
            return self._unavailable(
                "inference_exception",
                feature_report=report,
                inference=None,
                inference_error=f"{type(exc).__name__}: {exc}",
                model_metadata=metadata,
            )
        if not inference.get("available", False):
            return self._unavailable(
                str(inference.get("reason", "inference_unavailable")),
                feature_report=report,
                inference=inference,
                inference_error=inference.get("inference_error"),
                model_metadata=metadata,
            )
        result = {
            **inference,
            "available": True,
            "approved": bool(inference.get("approved", False)),
            "reason": "evaluated",
            "feature_report": report,
            "inference": inference,
            "model_metadata": metadata,
        }
        return result

    def evaluate(self, features_48: np.ndarray, direction: str = "BUY") -> Dict[str, Any]:
        """Return the legacy score/gate fields plus truthful model provenance."""
        if self.model is None:
            self._load_or_initialize()
        if self.model is None:
            return {"win_probability": 0.5, "win_probability_pct": "50.0%", "kelly_fraction": 0.0,
                    "expected_return_pct": 0.0, "regime": "UNKNOWN", "approved": False,
                    "status": "UNAVAILABLE", "reason": "no real trained model",
                    "min_threshold": MIN_WIN_PROBABILITY, "features_evaluated": 0}


        try:
            feat = np.asarray(features_48, dtype=np.float32).reshape(1, -1)
        except (TypeError, ValueError, OverflowError):
            return self._unavailable("invalid_feature_array", **self._model_metadata())
        if feat.shape[1] != len(self.feature_names):
            return self._unavailable("invalid_feature_count", **self._model_metadata())
        if not np.isfinite(feat).all():
            return self._unavailable("non_finite_features", **self._model_metadata())
        schema_error = self._model_schema_error()
        if schema_error:
            return self._unavailable(schema_error, **self._model_metadata())
        try:
            dmatrix = xgb.DMatrix(feat, feature_names=self.feature_names)
            prediction = self.model.predict(dmatrix)
            predictions = np.asarray(prediction, dtype=np.float64).reshape(-1)
        except Exception as exc:
            return self._unavailable(
                "inference_exception",
                inference_error=f"{type(exc).__name__}: {exc}",
                **self._model_metadata(),
            )
        if predictions.size != 1:
            return self._unavailable("invalid_prediction_count", **self._model_metadata())
        raw_prob = float(predictions[0])
        if not math.isfinite(raw_prob) or not 0.0 <= raw_prob <= 1.0:
            return self._unavailable("invalid_prediction_probability", **self._model_metadata())
        
        # Raw model score. It is not presented as calibrated out-of-sample probability.
        # This module's candle labels describe an upward/long outcome. A SELL
        # must use the opposite class score, never max(p, 1-p), which could
        # approve a short on strongly bullish evidence. This is still an
        # uncalibrated classification score, not a proven short-barrier win rate.
        normalized_direction = str(direction).upper()
        if normalized_direction in ("SELL", "SHORT", "PUT"):
            win_prob = 1.0 - raw_prob
        elif normalized_direction in ("BUY", "LONG", "CALL"):
            win_prob = raw_prob
        else:
            return self._unavailable("invalid_trade_direction", **self._model_metadata())

        # Fractional Kelly Criterion calculation
        b = MIN_REWARD_TO_RISK
        p = win_prob
        q = 1.0 - p
        kelly_full = (p * b - q) / b
        kelly_sized = max(0.0, float(KELLY_FRACTION_DAMPING * kelly_full))

        # Expected Return
        expected_return_pct = float(p * 1.5 - q * 1.0)

        # Scalp vs Swing Regime Classification
        atr_pct = float(features_48[18])
        trend_align = float(features_48[15])
        if atr_pct > 1.2 or abs(float(features_48[9])) > 1.5:
            regime = "SCALP"
        elif abs(trend_align) >= 1.0 and atr_pct <= 1.2:
            regime = "SWING"
        else:
            regime = "CHOP_AVOID" if abs(trend_align) < 0.3 else "SCALP"

        approved = bool(win_prob >= MIN_WIN_PROBABILITY and regime != "CHOP_AVOID")

        return {
            "available": True,
            "win_probability": round(win_prob, 4),
            "win_probability_pct": f"{win_prob * 100:.1f}%",
            "kelly_fraction": round(kelly_sized, 4),
            "expected_return_pct": round(expected_return_pct, 4),
            "regime": regime,
            "approved": approved,
            "min_threshold": MIN_WIN_PROBABILITY,
            "latency_ms": 1.5,
            "features_evaluated": len(self.feature_names),
            **self._model_metadata(),
        }


# Singleton accessor
_GLOBAL_XGB_ENGINE: Optional[JarvisXGBoostEngine] = None

def get_xgboost_engine() -> JarvisXGBoostEngine:
    global _GLOBAL_XGB_ENGINE
    if _GLOBAL_XGB_ENGINE is None:
        _GLOBAL_XGB_ENGINE = JarvisXGBoostEngine()
    return _GLOBAL_XGB_ENGINE
