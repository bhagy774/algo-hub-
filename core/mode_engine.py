"""Omni-timeframe decision engine with explicit Parts 1–12 integration.

Part 1's 13 and Part 2's 16 brains retain their existing votes. Parts 4–10
add seven independent engine votes (36 total). Part 3 retains its institutional
super-vote. Part 11 confirms direction; Part 12 gates confidence. The separate
Part 5 institutional override remains subordinate to risk/model gates.

Each provided nonempty frame is analyzed. XGBoost consumes actual named Part
results, with missing features reported rather than fabricated. No orders here.
"""
import time as _real_time
import numpy as np
import pandas as pd

import part1_FIXED as p1
import part2_FIXED as p2
import part3_FIXED as p3

import copy
import importlib
import math
from typing import Any



PART_CLASSES = {
    'part4_volume': ('part4_FIXED', 'VolumeProfileEngineGPU'),
    'part5_ml': ('part5_FIXED', 'MLEngineGPU'),
    'part6_trend': ('part6_FIXED', 'TrendEngineGPU'),
    'part7_volatility': ('part7_FIXED', 'VolatilityEngineGPU'),
    'part8_structure': ('part8_FIXED', 'MarketStructureEngineGPU'),
    'part9_orderflow': ('part9_FIXED', 'OrderflowEngineGPU'),
    'part10_candlestats': ('part10_FIXED', 'CandleStatsEngineGPU'),
    'part11_fusion': ('part11_FIXED', 'SignalFusionEngineGPU'),
    'part12_confidence': ('part12_FIXED', 'ConfidenceEngineGPU'),
}
VOTE_NAMES = {
    'part4_volume': 'P4_VolumeProfile', 'part5_ml': 'P5_StatisticalML',
    'part6_trend': 'P6_Trend', 'part7_volatility': 'P7_Volatility',
    'part8_structure': 'P8_MarketStructure', 'part9_orderflow': 'P9_Orderflow',
    'part10_candlestats': 'P10_CandleStats',
}


def direction(value: Any) -> str:
    if isinstance(value, str):
        named = {'BUY': 'BUY', 'CALL': 'BUY', 'SELL': 'SELL', 'PUT': 'SELL',
                 'NEUTRAL': 'NEUTRAL', 'NO TRADE': 'NEUTRAL', 'HOLD': 'NEUTRAL'}
        if value.upper() in named:
            return named[value.upper()]
    try:
        number = float(value)
        if not math.isfinite(number):
            return 'NEUTRAL'
        return 'BUY' if number > 0 else 'SELL' if number < 0 else 'NEUTRAL'
    except (ValueError, TypeError):
        return 'NEUTRAL'


class PartEngineBridge:
    def __init__(self, engines=None, *, importer=importlib.import_module):
        self.engines = {}
        self.initialization = {}
        if engines is not None:  # Explicit dependency injection for offline tests.
            self.engines.update(engines)
            for name in PART_CLASSES:
                self.initialization[name] = {'status': 'ready' if name in engines else 'unavailable'}
        else:
            for name, (module, cls) in PART_CLASSES.items():
                try:
                    self.engines[name] = getattr(importer(module), cls)()
                    self.initialization[name] = {'status': 'ready', 'class': cls}
                except Exception as error:
                    self.initialization[name] = {'status': 'unavailable', 'class': cls,
                                                 'error': f'{type(error).__name__}: {error}'}
        self.reset()

    def reset(self):
        self.frames = {}
        self.errors = {}

    def _analyze(self, name, data, context):
        engine = self.engines.get(name)
        error = None
        if engine is not None:
            try:
                # Prevent an analyzer from mutating another Part's evidence.
                payload = data.copy(deep=True) if name in VOTE_NAMES and hasattr(data, 'iloc') else copy.deepcopy(data)
                result = engine.analyze(payload, context=dict(context)) if name in VOTE_NAMES else engine.analyze(payload)
                if not isinstance(result, dict):
                    raise ValueError('analyzer returned non-dictionary')
                field = 'confidence' if name == 'part12_confidence' else 'signal'
                if field not in result:
                    raise ValueError(f'missing {field}')
                value = float(result[field])
                if not math.isfinite(value):
                    raise ValueError(f'non-finite {field}')
                if field == 'confidence' and not 0 <= value <= 100:
                    raise ValueError('confidence outside 0–100 scale')
                if field == 'signal' and value not in (-1, 0, 1):
                    raise ValueError('directional signal must be -1, 0, or 1')
                if 'confidence' in result:
                    confidence = float(result['confidence'])
                    if not math.isfinite(confidence) or not 0 <= confidence <= 100:
                        raise ValueError('invalid confidence')
                if 'telemetry' in result and not isinstance(result['telemetry'], dict):
                    raise ValueError('telemetry must be a dictionary')
                if name == 'part7_volatility' and result.get('telemetry', {}).get('status') in ('insufficient_or_invalid_ohlcv', 'analysis_error'):
                    self.errors.setdefault(context['timeframe'], {})[name] = result.get('thought', 'volatility unavailable')
                    return {**result, 'runtime_status': 'unavailable', 'runtime_source': PART_CLASSES[name][1]}
                return {**result, 'runtime_status': 'native', 'runtime_source': PART_CLASSES[name][1]}
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        else:
            error = self.initialization[name].get('error', 'engine unavailable')
        self.errors.setdefault(context['timeframe'], {})[name] = error
        if name in VOTE_NAMES:
            # Parts 4–6 and 8–10 delegate to these same mathematical functions.
            # Part 7 uses a different, explicitly labeled CPU risk fallback.
            # Neither path fabricates a successful native/GPU invocation.
            try:
                import quantitative_math as qm
                number = name.split('_', 1)[0][4:]
                return {**qm.part_signal(number, data, context=context), 'runtime_status': 'math_fallback',
                        'runtime_source': f'quantitative_math.part_signal({number})',
                        'native_error': error}
            except Exception as exc:
                error += f'; fallback {type(exc).__name__}: {exc}'
        return {'signal': 0, 'confidence': 0, 'runtime_status': 'unavailable',
                'runtime_source': PART_CLASSES[name][1], 'thought': 'engine unavailable', 'error': error}

    def evaluate_frame(self, timeframe, frame, seed_results, context=None):
        context = dict(context or {})
        context['timeframe'] = timeframe
        results = dict(seed_results)
        for name in VOTE_NAMES:
            results[name] = self._analyze(name, frame, context)
        # Named dictionaries preserve Part 11/12 weights and anchors. Part 12
        # evaluates Parts 1–10, not an extra copy of the Part 11 aggregate.
        inputs = dict(results)
        results['part11_fusion'] = self._analyze('part11_fusion', inputs, context)
        results['part12_confidence'] = self._analyze('part12_confidence', inputs, context)
        self.frames[timeframe] = results
        return results

    def select_frame(self, mode):
        preferred = ('1h', '60m', '4h', '240m', '1d') if mode == 'SWING' else ('5m', '5min', '1m', '3m', '15m')
        return next((tf for tf in preferred if tf in self.frames), next(iter(self.frames), None))

    def aggregate_track_results(self, mode):
        tf = self.select_frame(mode)
        if tf is None:
            return None, {}
        track_tfs = (
            ('30m', '1h', '60m', '2h', '120m', '4h', '240m', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
            if mode == 'SWING' else
            ('1m', '1min', '3m', '3min', '5m', '5min', '15m', '15min')
        )
        avail_tfs = [t for t in track_tfs if t in self.frames]
        if not avail_tfs:
            avail_tfs = [tf]
        base_results = dict(self.frames.get(tf, {}))
        part_keys = (
            'part1_breakout', 'part2_zone', 'part3_psychology', 'part4_volume',
            'part5_ml', 'part6_trend', 'part7_volatility', 'part8_structure',
            'part9_orderflow', 'part10_candlestats',
        )
        pooled = {}
        for pk in part_keys:
            base_item = dict(base_results.get(pk) or {'signal': 0, 'confidence': 0.0, 'thought': 'neutral'})
            if pk == 'part7_volatility':
                pooled[pk] = base_item
                continue
            buy_c = 0
            sell_c = 0
            best_conf = 0.0
            best_thought = base_item.get('thought', '')
            best_tele = dict(base_item.get('telemetry') or {})
            for t_name in avail_tfs:
                it = (self.frames.get(t_name) or {}).get(pk) or {}
                s_dir = direction(it.get('signal'))
                c_val = float(it.get('confidence', 0.0) or 0.0)
                if s_dir == 'BUY':
                    buy_c += 1
                    if c_val >= best_conf:
                        best_conf = c_val
                        best_thought = f"{it.get('thought', '')} [{t_name}]"
                        if isinstance(it.get('telemetry'), dict):
                            best_tele = dict(it['telemetry'])
                elif s_dir == 'SELL':
                    sell_c += 1
                    if c_val >= best_conf:
                        best_conf = c_val
                        best_thought = f"{it.get('thought', '')} [{t_name}]"
                        if isinstance(it.get('telemetry'), dict):
                            best_tele = dict(it['telemetry'])
                elif not best_tele and isinstance(it.get('telemetry'), dict):
                    best_tele = dict(it['telemetry'])
            agg_sig = 1 if buy_c > sell_c else (-1 if sell_c > buy_c else 0)
            base_item['signal'] = agg_sig
            if agg_sig != 0:
                base_item['confidence'] = max(best_conf, float(base_item.get('confidence', 0.0) or 0.0))
                base_item['thought'] = best_thought
            if best_tele:
                base_item['telemetry'] = best_tele
            pooled[pk] = base_item
        ctx = {'timeframe': tf, 'track_mode': mode, 'track_timeframes': avail_tfs}
        pooled['part11_fusion'] = self._analyze('part11_fusion', dict(pooled), ctx)
        pooled['part12_confidence'] = self._analyze('part12_confidence', dict(pooled), ctx)
        return tf, pooled

    def gate(self, mode, winning_direction, min_confidence=65.0):
        tf, results = self.aggregate_track_results(mode)
        if tf is None or not results:
            return 'PART_ENGINE_NO_DATA'
        for frame_tf, frame_results in self.frames.items():
            risk = frame_results.get('part7_volatility', {})
            telemetry = risk.get('telemetry', {})
            if risk.get('runtime_status') == 'unavailable' or telemetry.get('status') == 'insufficient_or_invalid_ohlcv':
                return 'PART7_VOLATILITY_UNAVAILABLE'
        entry_risk = self.frames.get(tf, {}).get('part7_volatility', {})
        entry_telemetry = entry_risk.get('telemetry', {}) if isinstance(entry_risk.get('telemetry'), dict) else {}
        if entry_risk.get('entry_blocked') or entry_risk.get('risk_veto') or entry_telemetry.get('entry_blocked') or entry_telemetry.get('risk_veto'):
            return 'PART7_VOLATILITY_VETO'
        fusion = results['part11_fusion']
        confidence = results['part12_confidence']
        if fusion.get('runtime_status') == 'unavailable':
            return 'PART11_FUSION_UNAVAILABLE'
        if direction(fusion.get('signal')) != winning_direction:
            return 'PART11_FUSION_NO_CONFIRMATION'
        if confidence.get('runtime_status') == 'unavailable':
            return 'PART12_CONFIDENCE_UNAVAILABLE'
        if float(confidence['confidence']) < min_confidence:
            return 'PART12_LOW_CONFIDENCE'
        return None

    def report(self, mode=None):
        tf, pooled = self.aggregate_track_results(mode)
        if not pooled and tf is not None:
            pooled = dict(self.frames.get(tf, {}))
        return {'initialization': dict(self.initialization), 'errors': dict(self.errors),
                'timeframe_results': dict(self.frames), 'selected_timeframe': tf,
                'selected_part_results': pooled,
                'roles': {'parts4_to10': 'directional_votes', 'part11': 'direction_confirmation',
                          'part12': 'confidence_gate', 'part7': 'advisory_volatility_regime'}}



# ------------------------------------------------------------------ simulated clock
class _SimClock:
    now = None

    def time(self):
        return self.now if self.now is not None else _real_time.time()

    def __getattr__(self, name):
        return getattr(_real_time, name)


CLOCK = _SimClock()
p2.time = CLOCK


# ------------------------------------------------------------------ helpers
def _p1_market_data(df):
    if df is None or df.empty:
        return {"price_action": [], "volume_pattern": []}
    return {
        "price_action": df[["open", "high", "low", "close"]].to_dict("records"),
        "volume_pattern": df["volume"].tolist(),
    }


def _breakout_strength(df, n=20):
    if df is None or len(df) < n + 1:
        return 0.2
    c, h, l = df["close"].iloc[-1], df["high"].iloc[-n - 1:-1].max(), df["low"].iloc[-n - 1:-1].min()
    return 0.6 if (c > h or c < l) else 0.2


class _Silent:
    def __enter__(self):
        import io, sys
        self._o, sys.stdout = sys.stdout, io.StringIO()

    def __exit__(self, *a):
        import sys
        sys.stdout = self._o


# ------------------------------------------------------------------ universal 29-brain engine
class ModeEngine:
    SWING_CONSENSUS_THRESHOLD = 5
    SCALP_CONSENSUS_THRESHOLD = 6

    def __init__(self, *, enable_extended_parts=True, require_xgboost=True,
                 min_part12_confidence=65.0, part_bridge=None, xgboost_engine=None):
        self.enable_extended_parts = bool(enable_extended_parts)
        self.require_xgboost = bool(require_xgboost)
        self.min_part12_confidence = float(min_part12_confidence)
        if not np.isfinite(self.min_part12_confidence) or not 0 <= self.min_part12_confidence <= 100:
            raise ValueError('Part 12 confidence threshold must be 0–100')
        self.xgboost_engine = xgboost_engine
        self.last_xgboost_report = {'available': False, 'approved': False, 'reason': 'not_evaluated'}
        if self.enable_extended_parts:
            
            self.part_bridge = part_bridge if part_bridge is not None else PartEngineBridge()
            # Total votes: 13 P1 + 16 P2 + 1 P3 + 7 Parts(4-10) = 37 independent votes.
            # Scale the old 29-brain thresholds proportionally: ceil(old_th * 37 / 29)
            self.SWING_CONSENSUS_THRESHOLD = 7   # ceil(5 * 37 / 29) = 7
            self.SCALP_CONSENSUS_THRESHOLD = 8   # ceil(6 * 37 / 29) = 8
        else:
            self.part_bridge = None
        with _Silent():
            self.p2_system = p2.AdvancedAnalysisSystem()
            self.p2_brains = self.p2_system.brains
            
        # Part 1 All 13 Brains
        self.p1_trend = p1.TrendBrain()
        self.p1_vol = p1.VolatilityBrain()
        self.p1_strength = p1.StrengthBrain()
        self.p1_risk = p1.RiskBrain()
        self.p1_reversal = p1.ReversalBrain()
        self.p1_regime = p1.RegimeBrain()
        self.p1_deepseek = p1.DeepSeekBrain()
        self.p1_evolution = p1.EvolutionBrain()
        self.p1_memory = p1.MemoryBrain()
        self.p1_self_healing = p1.SelfHealingBrain()
        self.p1_meta_fusion = p1.MetaFusionBrain()
        self.p1_r1 = p1.MiniR1Brain()
        self.p1_v3 = p1.MiniV3Brain()
        # Part 3 Institutional Engine
        try:
            self.p3_institutional = p3.InstitutionalTradingEngineGPU()
        except Exception:
            self.p3_institutional = None
            
        # Part 5 Institutional Fusion Veto Engine
        try:
            import part5_FIXED as p5
            self.p5_fusion = p5.InstitutionalFusionEngine()
        except Exception:
            self.p5_fusion = None
            
        # Wire the Global Cognitive Bus to ALL Brains (Part 1, 2, and 3)
        try:
            from jarvis_cognitive_bus import CognitiveBus
            bus = CognitiveBus()
        except ImportError:
            bus = None

        if bus:
            self.p2_system.bus = bus
            for p2_brain in self.p2_brains.values():
                p2_brain.bus = bus
                
            p1_brains = [
                self.p1_trend, self.p1_vol, self.p1_strength, self.p1_risk, 
                self.p1_reversal, self.p1_regime, self.p1_deepseek, self.p1_evolution, 
                self.p1_memory, self.p1_self_healing, self.p1_meta_fusion, self.p1_r1, self.p1_v3
            ]
            for b in p1_brains:
                b.bus = bus
                
            if self.p3_institutional:
                self.p3_institutional.bus = bus

    # ---------------- XGBoost micro-brain support (SHADOW by default) ----------------
    # Each brain may expose an XGBoost confidence (0..1, 0.5 = neutral / no model).
    # By default it is only REPORTED (self.last_xgb_report). Vote weighting is OFF unless the
    # operator sets JARVIS_XGB_VOTE_WEIGHTING=1, and then ONLY brains whose own validated model
    # is active are weighted: weight = clamp(0.5 + confidence, 0.5, 1.5). Missing/neutral = 1.0.
    # This never changes the Part 5 veto, institutional rules or the global live_godmode veto.
    @staticmethod
    def _xgb_weighting_enabled():
        import os
        return os.environ.get("JARVIS_XGB_VOTE_WEIGHTING", "0") == "1"

    def _xgb_vote_weight(self, brain_name):
        if not self._xgb_weighting_enabled():
            return 1
        rep_ = getattr(self, "last_xgb_report", {}).get(brain_name)
        if not rep_ or not rep_.get("active"):
            return 1
        try:
            c = float(rep_.get("confidence", 0.5))
            if not (0.0 <= c <= 1.0):
                return 1
            return min(max(0.5 + c, 0.5), 1.5)
        except Exception:
            return 1

    @staticmethod
    def _xgb_from(obj):
        """(confidence, active) from a result dict or a brain's last_xgb side-channel."""
        try:
            if isinstance(obj, dict) and "xgb_confidence" in obj:
                return float(obj["xgb_confidence"]), bool(obj.get("xgb_active", False))
            lx = getattr(obj, "last_xgb", None)
            if isinstance(lx, dict):
                return float(lx.get("confidence", 0.5)), bool(lx.get("active", False))
        except Exception:
            pass
        return None

    def evaluate_all_brains(self, tf_dict, context=None):
        """
        Omni-Timeframe Loop:
        EVERY SINGLE ONE of the 29 Brains evaluates ALL available timeframes (1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d).
        """
        if self.part_bridge is not None:
            self.part_bridge.reset()
        available_tfs = {}
        for k, v in tf_dict.items():
            if v is not None and not v.empty:
                available_tfs[k] = v

        primary_df = list(available_tfs.values())[0] if available_tfs else None
        price = primary_df['close'].iloc[-1] if primary_df is not None else 0
        
        brain_tf_signals = {}  # {brain_name: {tf_name: (direction, confidence)}}
        xgb_acc = {}  # {brain_name: [(confidence, active), ...]} (shadow report)

        # Iterate through EVERY timeframe provided by API
        for tf_name, df in available_tfs.items():
            md = _p1_market_data(df)
            cndl = df.iloc[-1].to_dict() if not df.empty else {}
            
            # --- Part 1 Brains on this specific TF ---
            t_res = self.p1_trend.analyze_trend(md)
            v_res = self.p1_vol.analyze_volatility(md)
            brk_res = {"strength": _breakout_strength(df)}
            st_res = self.p1_strength.analyze_strength(md, brk_res, {})
            rk_res = self.p1_risk.analyze_risk(md, v_res, t_res)
            rv_res = self.p1_reversal.analyze_reversal(md, t_res, st_res)
            reg_res = self.p1_regime.analyze_regime(md, t_res, v_res)
            ds_res = self.p1_deepseek.analyze_deepseek(md, {"trend": t_res, "strength": st_res, "risk": rk_res, "reversal": rv_res})
            evo_res = self.p1_evolution.analyze_evolution(md, [])
            mem_res = self.p1_memory.analyze_memory(md, None)
            heal_res = self.p1_self_healing.analyze_self_heal(md, {})
            r1_res = self.p1_r1.analyze_mini_r1(md, brk_res, rk_res)
            v3_res = self.p1_v3.analyze_mini_v3(md)
            mf_res = self.p1_meta_fusion.analyze_meta_fusion({
                'trend': t_res,
                'volatility': v_res,
                'strength': st_res,
                'risk': rk_res,
                'reversal': rv_res,
                'regime': reg_res,
                'deepseek': ds_res,
                'evolution': evo_res,
                'memory': mem_res,
                'self_healing': heal_res,
                'mini_r1': r1_res,
                'mini_v3': v3_res
            })

            for _n, _o in (("P1_Trend", t_res), ("P1_Volatility", v_res), ("P1_Strength", st_res),
                           ("P1_Risk", rk_res), ("P1_Reversal", rv_res), ("P1_Regime", reg_res),
                           ("P1_DeepSeek", ds_res), ("P1_Evolution", evo_res), ("P1_Memory", mem_res),
                           ("P1_SelfHealing", heal_res), ("P1_MiniR1", r1_res), ("P1_MiniV3", v3_res),
                           ("P1_MetaFusion", mf_res)):
                _x = self._xgb_from(_o)
                if _x:
                    xgb_acc.setdefault(_n, []).append(_x)

            def add_tf_sig(b_name, dir_val, conf_val):
                if b_name not in brain_tf_signals: brain_tf_signals[b_name] = {}
                brain_tf_signals[b_name][tf_name] = (dir_val, conf_val)

            add_tf_sig("P1_Trend", "BUY" if t_res["support_score"] > 0.05 else "SELL" if t_res["support_score"] < -0.05 else "NEUTRAL", abs(t_res["support_score"]))
            add_tf_sig("P1_Volatility", "BUY" if v_res.get("volatility_regime") == "HIGH" and t_res["trend_direction"] == 1 else "SELL" if v_res.get("volatility_regime") == "HIGH" else "NEUTRAL", 0.7)
            add_tf_sig("P1_Strength", "BUY" if st_res["overall_strength"] > 0.3 and t_res["trend_direction"] == 1 else "SELL" if st_res["overall_strength"] > 0.3 else "NEUTRAL", st_res["overall_strength"])
            # risk_score is 0..1 where LOW risk (< 0.4) means safer to trade in trend direction
            _rk_score = float(rk_res.get("risk_score", 1.0))
            add_tf_sig("P1_Risk", "BUY" if _rk_score < 0.4 and t_res["trend_direction"] == 1 else "SELL" if _rk_score < 0.4 and t_res["trend_direction"] == -1 else "NEUTRAL", 1.0 - _rk_score)
            # reversal: if uptrend + reversal detected => SELL (market will reverse down)
            #           if downtrend + reversal detected => BUY (market will reverse up)
            _rv_prob = float(rv_res.get("reversal_probability", 0))
            add_tf_sig("P1_Reversal", "SELL" if _rv_prob > 0.001 and t_res["trend_direction"] == 1 else "BUY" if _rv_prob > 0.001 and t_res["trend_direction"] == -1 else "NEUTRAL", _rv_prob)
            add_tf_sig("P1_Regime", "BUY" if "BULL" in reg_res.get("regime_type", "") else "SELL" if "BEAR" in reg_res.get("regime_type", "") else "NEUTRAL", reg_res.get("regime_confidence", 0.5))
            # DeepSeek: fakeout_block is fakeout risk; low risk + direction = trade signal
            _ds_block = float(ds_res.get("fakeout_block", 0))
            add_tf_sig("P1_DeepSeek", "BUY" if _ds_block < 0.3 and t_res["trend_direction"] == 1 else "SELL" if _ds_block < 0.3 and t_res["trend_direction"] == -1 else "NEUTRAL", 1.0 - _ds_block)
            # Evolution: support_score > 0.5 with bullish = BUY, bearish = SELL
            _evo_sc = float(evo_res.get("support_score", 0))
            add_tf_sig("P1_Evolution", "BUY" if _evo_sc > 0.5 and t_res["trend_direction"] == 1 else "SELL" if _evo_sc > 0.5 and t_res["trend_direction"] == -1 else "NEUTRAL", _evo_sc)
            add_tf_sig("P1_Memory", "BUY" if mem_res.get("support_score", 0) > 0.1 else "SELL" if mem_res.get("support_score", 0) < -0.1 else "NEUTRAL", abs(mem_res.get("support_score", 0)))
            # SelfHealing: health_index > 0.6 means system is healthy enough to trade in trend direction
            _hi = float(heal_res.get("health_index", 0))
            add_tf_sig("P1_SelfHealing", "BUY" if _hi > 0.6 and t_res["trend_direction"] == 1 else "SELL" if _hi > 0.6 and t_res["trend_direction"] == -1 else "NEUTRAL", _hi)
            # MiniR1: trap_probability < 0.4 means market is NOT a trap; safe to trade in trend direction
            _r1_trap = float(r1_res.get("trap_probability", 0))
            add_tf_sig("P1_MiniR1", "BUY" if _r1_trap < 0.4 and t_res["trend_direction"] == 1 else "SELL" if _r1_trap < 0.4 and t_res["trend_direction"] == -1 else "NEUTRAL", 1.0 - _r1_trap)
            add_tf_sig("P1_MiniV3", "BUY" if v3_res.get("tick_momentum", 0) > 0.15 else "SELL" if v3_res.get("tick_momentum", 0) < -0.15 else "NEUTRAL", abs(v3_res.get("tick_momentum", 0)))
            # MetaFusion: support_score > 0.1 with direction = trade signal
            _mf_sc = float(mf_res.get("support_score", 0))
            add_tf_sig("P1_MetaFusion", "BUY" if _mf_sc > 0.1 and t_res["trend_direction"] == 1 else "SELL" if _mf_sc > 0.1 and t_res["trend_direction"] == -1 else "NEUTRAL", abs(_mf_sc))

            # --- Part 2 Brains on this specific TF ---
            b2 = self.p2_brains
            psy = b2["candle_psychology"].analyze_candle_psychology(cndl)

            def eval_p2_tf(fn):
                try:
                    sigs = fn() or []
                    if isinstance(sigs, dict): sigs = list(sigs.items())
                    call_s = sum(s[1] for s in sigs if isinstance(s, (list, tuple)) and len(s) >= 2 and str(s[0]).upper() in ("CALL", "BUY"))
                    put_s = sum(s[1] for s in sigs if isinstance(s, (list, tuple)) and len(s) >= 2 and str(s[0]).upper() in ("PUT", "SELL"))
                    if call_s > put_s and call_s > 4.0: return ("BUY", call_s)
                    elif put_s > call_s and put_s > 4.0: return ("SELL", put_s)
                except Exception: pass
                return ("NEUTRAL", 0)

            # Pass the correct per-timeframe OHLCV data to zone detector (not same df 3x)
            _z_1m = available_tfs.get('1m', available_tfs.get('1min', df))
            _z_5m = available_tfs.get('5m', available_tfs.get('5min', df))
            _z_15m = available_tfs.get('15m', available_tfs.get('15min', df))
            add_tf_sig("P2_ZonePointFiveDetector", *eval_p2_tf(lambda: b2["zone_detector"].detect_0_5_zone_signals(cndl, psy, _z_1m, _z_5m, _z_15m)))
            add_tf_sig("P2_CandlePsychologyMaster", "BUY" if psy.get("is_bullish") else "SELL" if psy.get("is_bearish") else "NEUTRAL", 7.0)
            add_tf_sig("P2_VolumeProfile", *eval_p2_tf(lambda: (b2["volume_profile"].analyze_volume_profile(df, tf_name), b2["volume_profile"].detect_volume_signals(cndl, psy, df))[1]))
            add_tf_sig("P2_MarketStructure", *eval_p2_tf(lambda: (b2["market_structure"].analyze_market_structure(df, tf_name), b2["market_structure"].get_structure_signals(price, psy))[1]))
            add_tf_sig("P2_OrderFlow", *eval_p2_tf(lambda: (b2["order_flow"].analyze_order_flow(df), b2["order_flow"].get_orderflow_signals(price, psy, b2["volume_profile"].detect_volume_spike(df)))[1]))
            add_tf_sig("P2_MomentumOscillator", *eval_p2_tf(lambda: (b2["momentum_oscillator"].calculate_all_oscillators(df), b2["momentum_oscillator"].get_momentum_signals(price, psy))[1]))
            add_tf_sig("P2_VolatilityRegime", *eval_p2_tf(lambda: (b2["volatility_regime"].analyze_volatility_regime(df), b2["volatility_regime"].get_volatility_signals(price, psy))[1]))
            add_tf_sig("P2_CycleAnalysis", *eval_p2_tf(lambda: (b2["cycle_analysis"].analyze_market_cycles(df), b2["cycle_analysis"].get_cycle_signals(price, psy))[1]))
            add_tf_sig("P2_CorrelationMatrix", *eval_p2_tf(lambda: b2["correlation_matrix"].get_correlation_signals(price, psy)))
            add_tf_sig("P2_PatternRecognition", *eval_p2_tf(lambda: (b2["pattern_recognition"].recognize_chart_patterns(df), b2["pattern_recognition"].get_pattern_signals(price, psy))[1]))
            add_tf_sig("P2_SupportResistance", *eval_p2_tf(lambda: (b2["support_resistance"].calculate_support_resistance(df), b2["support_resistance"].get_sr_signals(price, psy))[1]))
            add_tf_sig("P2_TrendAnalysis", *eval_p2_tf(lambda: (al := b2["trend_analysis"].analyze_multi_timeframe_trends(tf_dict), b2["trend_analysis"].get_trend_signals(price, psy, al.get("alignment", "NEUTRAL")))[1]))
            # detect_market_regime() returns the regime STRING which must be passed to get_regime_signals
            add_tf_sig("P2_MarketRegime", *eval_p2_tf(lambda: (lambda r: b2["market_regime"].get_regime_signals(price, psy, r))(b2["market_regime"].detect_market_regime(df))))
            add_tf_sig("P2_PriceAction", *eval_p2_tf(lambda: (b2["price_action"].analyze_price_action(df), b2["price_action"].get_pa_signals(price, psy))[1]))
            add_tf_sig("P2_InstitutionalFlow", *eval_p2_tf(lambda: (b2["institutional_flow"].analyze_institutional_flow(df), b2["institutional_flow"].get_institutional_signals(price, psy))[1]))
            add_tf_sig("P2_SignalFusion", *eval_p2_tf(lambda: b2["signal_fusion"].fuse_signals([], price, {})))

            # Parts 4–12 use the same native OHLCV frame, not aggregate vote
            # counts disguised as candle input. Parts 11/12 get named results.
            if self.part_bridge is not None:
                
                meta_score = float(mf_res.get('support_score', 0.0))
                # p1_dir must reflect actual trend direction, not just score magnitude
                p1_dir = 1 if meta_score > 0.1 and t_res['trend_direction'] == 1 else -1 if meta_score > 0.1 and t_res['trend_direction'] == -1 else 0
                zone_dir = brain_tf_signals['P2_ZonePointFiveDetector'][tf_name][0]
                seed = {
                    'part1_breakout': {'signal': p1_dir, 'thought': 'ModeEngine Part 1 MetaFusion summary',
                                       'source': 'P1_MetaFusion', 'raw_result': mf_res},
                    'part2_zone': {'signal': 1 if zone_dir == 'BUY' else -1 if zone_dir == 'SELL' else 0,
                                   'thought': 'ModeEngine Part 2 native zone signal', 'source': 'P2_ZonePointFiveDetector'},
                    'part3_psychology': {'signal': 0, 'thought': 'institutional engine unavailable',
                                        'source': 'Part3_Institutional', 'runtime_status': 'unavailable'},
                }
                frame_context = dict(context or {})
                if self.p3_institutional is not None:
                    try:
                        p3_frame = self.p3_institutional.generate_mtf_signals({tf_name: df})
                        consensus = float(p3_frame.get('mtf_consensus', 0.0))
                        if not np.isfinite(consensus):
                            raise ValueError('non-finite institutional consensus')
                        components = p3_frame.get('components', {})
                        if isinstance(components, dict):
                            external_components = frame_context.get('institutional_components', {})
                            if not isinstance(external_components, dict):
                                external_components = {}
                            frame_context['institutional_components'] = {
                                **external_components, **components,
                                # A supplied neutral regime must not erase a
                                # native PANIC/VOLATILE observation, or vice versa.
                                'regime': f"{components.get('regime', '')} {external_components.get('regime', '')}".strip(),
                            }
                        seed['part3_psychology'] = {
                            'signal': 1 if consensus >= 0.25 else -1 if consensus <= -0.25 else 0,
                            'thought': 'ModeEngine Part 3 institutional frame consensus',
                            'source': 'Part3_Institutional', 'raw_result': p3_frame,
                        }
                    except Exception as error:
                        seed['part3_psychology']['error'] = f'{type(error).__name__}: {error}'
                frame_results = self.part_bridge.evaluate_frame(tf_name, df, seed, context=frame_context)
                for key, vote_name in VOTE_NAMES.items():
                    result = frame_results[key]
                    add_tf_sig(vote_name, direction(result.get('signal')), float(result.get('confidence', 0)))
                # Add P3_Institutional to brain_tf_signals for vote transparency
                p3_seed = seed.get('part3_psychology', {})
                _p3_sig = p3_seed.get('signal', 0)
                add_tf_sig('P3_Institutional', 'BUY' if _p3_sig == 1 else 'SELL' if _p3_sig == -1 else 'NEUTRAL', abs(float(_p3_sig)))
        # ---- shadow XGBoost report (Part 2/3 brains expose brain.last_xgb) ----
        _p2_map = {"P2_ZonePointFiveDetector": "zone_detector", "P2_CandlePsychologyMaster": "candle_psychology",
                   "P2_VolumeProfile": "volume_profile", "P2_MarketStructure": "market_structure",
                   "P2_OrderFlow": "order_flow", "P2_MomentumOscillator": "momentum_oscillator",
                   "P2_VolatilityRegime": "volatility_regime", "P2_CycleAnalysis": "cycle_analysis",
                   "P2_CorrelationMatrix": "correlation_matrix", "P2_PatternRecognition": "pattern_recognition",
                   "P2_SupportResistance": "support_resistance", "P2_TrendAnalysis": "trend_analysis",
                   "P2_MarketRegime": "market_regime", "P2_PriceAction": "price_action",
                   "P2_InstitutionalFlow": "institutional_flow", "P2_SignalFusion": "signal_fusion"}
        for _n, _k in _p2_map.items():
            _b = self.p2_brains.get(_k) if isinstance(self.p2_brains, dict) else None
            _x = self._xgb_from(_b) if _b is not None else None
            if _x:
                xgb_acc.setdefault(_n, []).append(_x)
        _p3 = getattr(self, "p3_institutional", None)
        _x = self._xgb_from(_p3) if _p3 is not None else None
        if _x:
            xgb_acc.setdefault("P3_Institutional", []).append(_x)
        _report = {}
        for _n, _l in xgb_acc.items():
            _act = [c for c, a in _l if a]
            _report[_n] = {"confidence": float(sum(_act) / len(_act)) if _act else 0.5,
                           "active": bool(_act)}
        self.last_xgb_report = _report

        # Synthesize final vote per brain separating SCALP (1m-30m) vs SWING (1h+)
        final_brain_votes = {}
        scalp_tfs = {"1m", "3m", "5m", "15m", "1min", "3min", "5min", "15min"}
        swing_tfs = {"30m", "30min", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M", "60m", "120m", "240m"}

        for b_name, tf_map in brain_tf_signals.items():
            scalp_buy = sum(1 for tf, (d, _) in tf_map.items() if d == "BUY" and tf in scalp_tfs)
            scalp_sell = sum(1 for tf, (d, _) in tf_map.items() if d == "SELL" and tf in scalp_tfs)
            swing_buy = sum(1 for tf, (d, _) in tf_map.items() if d == "BUY" and tf in swing_tfs)
            swing_sell = sum(1 for tf, (d, _) in tf_map.items() if d == "SELL" and tf in swing_tfs)

            scalp_net = scalp_buy - scalp_sell
            swing_net = swing_buy - swing_sell

            if swing_net == 0 and scalp_net == 0:
                final_brain_votes[b_name] = ("NEUTRAL", "NEUTRAL")
            elif abs(swing_net) > abs(scalp_net) or (abs(swing_net) == abs(scalp_net) and swing_net != 0 and (swing_net * scalp_net > 0 or scalp_net == 0)):
                final_brain_votes[b_name] = ("SWING", "BUY" if swing_net > 0 else "SELL")
            elif abs(scalp_net) > abs(swing_net):
                final_brain_votes[b_name] = ("SCALP", "BUY" if scalp_net > 0 else "SELL")
            else:
                # Equal opposing strength between scalp and swing buckets -> neutral
                final_brain_votes[b_name] = ("NEUTRAL", "NEUTRAL")

        return final_brain_votes

    def calculate_dynamic_sl_tp(self, df, price, direction, trade_type="SCALP"):
        """Calculates dynamic SL/TP using SupportResistanceBrainGPU + ATR Guard."""
        b = self.p2_brains
        try:
            b["support_resistance"].calculate_support_resistance(df)
            levels = b["support_resistance"].get_nearest_levels(price)
        except Exception:
            levels = {'support': price * 0.99, 'resistance': price * 1.01}
            
        md = _p1_market_data(df)
        t = self.p1_trend.analyze_trend(md)
        v = self.p1_vol.analyze_volatility(md)
        rk = self.p1_risk.analyze_risk(md, v, t)
        atr_stop = max(float(rk.get("stop_distance", price * 0.003)), price * 0.0015)
        
        if direction == "BUY":
            sr_sl = levels['support'] * 0.998
            sl = min(sr_sl, price - (atr_stop * (2.5 if trade_type == "SWING" else 1.2)))
            sl_pct = (price - sl) / price
            
            min_rr = 2.0 if trade_type == "SWING" else 1.5
            sr_tp = max(levels['resistance'] * 1.002, price + (price - sl) * min_rr)
            tp = sr_tp
            tp_pct = (tp - price) / price
        else: # SELL
            sr_sl = levels['resistance'] * 1.002
            sl = max(sr_sl, price + (atr_stop * (2.5 if trade_type == "SWING" else 1.2)))
            sl_pct = (sl - price) / price
            
            min_rr = 2.0 if trade_type == "SWING" else 1.5
            sr_tp = min(levels['support'] * 0.998, price - (sl - price) * min_rr)
            tp = sr_tp
            tp_pct = (price - tp) / price
            
        rr = tp_pct / max(sl_pct, 1e-8)
        return round(sl, 2), round(tp, 2), round(rr, 2)

    @staticmethod
    def _prepare_inputs(m5, h1, tf_dict):
        required = ['open', 'high', 'low', 'close', 'volume']
        def validate(frame, name):
            if frame is None:
                return None
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                raise ValueError(f'{name}: empty or non-DataFrame input')
            if not set(required).issubset(frame.columns):
                raise ValueError(f'{name}: missing OHLCV columns')
            normalized = frame.copy(deep=True)
            try:
                normalized[required] = normalized[required].astype(float)
                values = normalized[required].to_numpy()
            except (ValueError, TypeError) as error:
                raise ValueError(f'{name}: nonnumeric OHLCV') from error
            if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
                raise ValueError(f'{name}: nonfinite/nonpositive prices or invalid volume')
            if (normalized['high'] < normalized[['open', 'close', 'low']].max(axis=1)).any() or (normalized['low'] > normalized[['open', 'close', 'high']].min(axis=1)).any():
                raise ValueError(f'{name}: invalid candle geometry')
            return normalized
        m5 = validate(m5, 'm5')
        h1 = validate(h1, 'h1')
        if tf_dict is None:
            tf_dict = {'5min': m5, '1h': h1}
        if not isinstance(tf_dict, dict):
            raise ValueError('timeframes must be a dictionary')
        frames = {str(tf): validate(df, str(tf)) for tf, df in tf_dict.items() if df is not None}
        if not frames:
            raise ValueError('no usable OHLCV timeframes')
        # The provided timeframe map is the decision evidence source of truth.
        # Resolve execution anchors from it before using optional legacy args.
        m5 = next((frames[tf] for tf in ('5m', '5min', '1m', '3m') if tf in frames), m5)
        h1 = next((frames[tf] for tf in ('1h', '60m', '4h', '240m', '1d') if tf in frames), h1)
        return m5, h1, frames

    def master_evaluate(self, m5, h1, tf_dict=None, *, snapshot=None, options_data=None):
        """Evaluate Parts 1–12 and attach auditable frame/model diagnostics.

        Extended mode defaults to fail-closed when real XGBoost inference is
        unavailable. Set require_xgboost=False explicitly for algorithm-only
        operation. This class never submits an order.
        """
        self.last_xgboost_report = {'available': False, 'approved': False, 'reason': 'no_candidate'}
        self._evaluation_snapshot = dict(snapshot or {})
        self._evaluation_options = dict(options_data or {})
        self._candidate_mode = None
        if self.part_bridge is not None:
            self.part_bridge.reset()
        try:
            m5, h1, tf_dict = self._prepare_inputs(m5, h1, tf_dict)
        except ValueError as error:
            result = {'action': 'HOLD', 'trade_type': None, 'direction': None,
                      'entry': None, 'sl': None, 'tp': None, 'rr': None,
                      'total_brains_voted': 0, 'timeframes_evaluated': [],
                      'all_brain_votes': {}, 'vote_tally': {}, 'reason': f'INVALID_MARKET_DATA: {error}'}
            self.last_xgboost_report = {'available': False, 'approved': False, 'reason': 'invalid_market_data'}
        else:
            result = self._master_evaluate(m5, h1, tf_dict)
        result['extended_parts_enabled'] = self.enable_extended_parts
        result['consensus_thresholds'] = {'SWING': self.SWING_CONSENSUS_THRESHOLD,
                                          'SCALP': self.SCALP_CONSENSUS_THRESHOLD}
        result['xgboost_report'] = self.last_xgboost_report
        if self.part_bridge is not None:
            report = self.part_bridge.report(result.get('trade_type') or getattr(self, '_candidate_mode', None))
            result['part_engine_report'] = report
            result['part12_confidence'] = report['selected_part_results'].get('part12_confidence', {}).get('confidence')
        return result

    def _master_evaluate(self, m5, h1, tf_dict=None):
        """
        Universal Master Evaluator:
        Tallies votes from ALL 29 BRAINS evaluated across ALL API TIMEFRAMES (1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d).
        """
        if tf_dict is None:
            tf_dict = {"5min": m5, "1h": h1}
            
        self._candidate_mode = None
        entry_frame = m5 if m5 is not None else h1
        if entry_frame is None:
            entry_frame = next(iter(tf_dict.values()))
        price = float(entry_frame['close'].iloc[-1])
        votes = self.evaluate_all_brains(tf_dict, context=self._evaluation_snapshot)

        tally = {
            ("SWING", "BUY"): 0,
            ("SWING", "SELL"): 0,
            ("SCALP", "BUY"): 0,
            ("SCALP", "SELL"): 0,
        }
        
        for brain_name, (m, d) in votes.items():
            if (m, d) in tally:
                tally[(m, d)] += self._xgb_vote_weight(brain_name)  # 1.0 unless opt-in weighting + active model

        # --- PART 3 INSTITUTIONAL SUPER-VOTING ---
        institutional_veto_dir = None
        if getattr(self, 'p3_institutional', None) is not None:
            try:
                p3_res = self.p3_institutional.generate_mtf_signals(tf_dict)
                p3_consensus = p3_res.get('mtf_consensus', 0.0)
                p3_regime = p3_res.get('components', {}).get('regime', 'UNKNOWN')
                
                if abs(p3_consensus) >= 0.25:
                    p3_dir = "BUY" if p3_consensus > 0 else "SELL"
                    p3_mode = "SWING" if "TRENDING" in str(p3_regime) else "SCALP"
                    institutional_veto_dir = p3_dir  # Save institutional direction to prevent counter-scalping
                    
                    if (p3_mode, p3_dir) in tally:
                        tally[(p3_mode, p3_dir)] += 3  # The Institutional Super-Vote
            except Exception:
                pass
        # --- END PART 3 ---

        # --- PART 5 EMERGENCY VETO (BLACK SWAN CATCHER) ---
        part5_veto = False
        if getattr(self, 'p5_fusion', None) is not None:
            try:
                m1 = tf_dict.get('1m')
                m5_data = tf_dict.get('5m', tf_dict.get('5min'))
                m15 = tf_dict.get('15m')
                if m1 is not None and m5_data is not None:
                    p5_res = self.p5_fusion.process_market_data(m1, m5_data, m15)
                    _p5_raw = p5_res.get('fused_signal', 'NO TRADE')
                    # InstitutionalFusionEngine returns CALL/PUT/NO TRADE; normalize to BUY/SELL
                    _p5_map = {'CALL': 'BUY', 'BUY': 'BUY', 'PUT': 'SELL', 'SELL': 'SELL'}
                    p5_signal = _p5_map.get(str(_p5_raw).upper())
                    if p5_signal in ["BUY", "SELL"]:
                        # Market is crashing/spiking - WIPE ALL NORMAL VOTES!
                        tally = {k: 0 for k in tally}
                        tally[("SWING", p5_signal)] = 100  # God-Mode VETO
                        tally[("SCALP", p5_signal)] = 100
                        part5_veto = True
            except Exception:
                pass
        # --- END PART 5 VETO ---

        swing_buy = tally[("SWING", "BUY")]
        swing_sell = tally[("SWING", "SELL")]
        scalp_buy = tally[("SCALP", "BUY")]
        scalp_sell = tally[("SCALP", "SELL")]

        # Phase 10: evaluate SWING and SCALP independently so a valid SCALP candidate is never lost when SWING also qualifies.
        def _cand(buy, sell, thr):
            if buy >= thr and buy > sell:
                return 'BUY'
            if sell >= thr and sell > buy:
                return 'SELL'
            return None
        self.last_dual_candidates = {
            'swing_candidate': _cand(swing_buy, swing_sell, self.SWING_CONSENSUS_THRESHOLD),
            'scalp_candidate': _cand(scalp_buy, scalp_sell, self.SCALP_CONSENSUS_THRESHOLD),
            'tally': {'SWING': [swing_buy, swing_sell], 'SCALP': [scalp_buy, scalp_sell]},
        }
        try:
            import logging as _lg
            _lg.getLogger(__name__).info('[MODE_CANDIDATES] swing=%s scalp=%s tally=%s', self.last_dual_candidates['swing_candidate'], self.last_dual_candidates['scalp_candidate'], self.last_dual_candidates['tally'])
        except Exception:
            pass
        winning_mode = None
        winning_dir = None

        _sc = self.last_dual_candidates['swing_candidate']
        _kc = self.last_dual_candidates['scalp_candidate']
        if _sc and _kc and _sc != _kc:
            winning_mode, winning_dir = None, None  # Phase 10: opposing SWING vs SCALP candidates -> no trade
        elif _sc and _kc:
            _smargin = abs(swing_buy - swing_sell) / max(self.SWING_CONSENSUS_THRESHOLD, 1)
            _kmargin = abs(scalp_buy - scalp_sell) / max(self.SCALP_CONSENSUS_THRESHOLD, 1)
            winning_mode = "SWING" if _smargin > _kmargin else "SCALP"
            winning_dir = _sc
        elif _sc:
            winning_mode, winning_dir = "SWING", _sc
        elif scalp_buy >= self.SCALP_CONSENSUS_THRESHOLD and scalp_buy > scalp_sell:
            winning_mode, winning_dir = "SCALP", "BUY"
        elif scalp_sell >= self.SCALP_CONSENSUS_THRESHOLD and scalp_sell > scalp_buy:
            winning_mode, winning_dir = "SCALP", "SELL"

        # --- INSTITUTIONAL ANTI-COUNTER-SCALP RULE ---
        if winning_mode == "SCALP" and institutional_veto_dir:
            if winning_dir != institutional_veto_dir:
                return {
                    "action": "HOLD",
                    "trade_type": None,
                    "direction": None,
                    "entry": price,
                    "sl": None,
                    "tp": None,
                    "rr": None,
                    "total_brains_voted": len(votes),
                    "timeframes_evaluated": list(tf_dict.keys()),
                    "all_brain_votes": votes,
                    "vote_tally": tally,
                    "reason": f"INSTITUTIONAL_VETO: Refusing to SCALP {winning_dir} against Institutional {institutional_veto_dir}"
                }
        # --- END VETO ---

        self._candidate_mode = winning_mode
        # Native Part 11 and 12 are confirmations, not extra copies of the
        # same evidence in the vote denominator. Part 7 risk outranks the
        # legacy Part 5 override as well.
        def extended_hold(reason):
            return {'action': 'HOLD', 'trade_type': None, 'direction': None,
                    'entry': price, 'sl': None, 'tp': None, 'rr': None,
                    'total_brains_voted': len(votes), 'timeframes_evaluated': list(tf_dict),
                    'all_brain_votes': votes, 'vote_tally': tally, 'reason': reason}

        if winning_dir and (m5 if winning_mode == 'SCALP' else h1) is None:
            return extended_hold('MISSING_EXECUTION_TIMEFRAME')

        if winning_dir and self.part_bridge is not None:
            gate_reason = self.part_bridge.gate(winning_mode, winning_dir, self.min_part12_confidence)
            if gate_reason:
                self.last_xgboost_report = {'available': False, 'approved': False,
                                            'reason': 'deterministic_part_gate_blocked'}
                return extended_hold(gate_reason)
            try:
                if self.xgboost_engine is None:
                    from jarvis_xgboost_engine import JarvisXGBoostEngine
                    # Never authorize trades with a synthetic on-the-fly model.
                    self.xgboost_engine = JarvisXGBoostEngine(allow_bootstrap=False)
                report = self.part_bridge.report(winning_mode)
                self.last_xgboost_report = self.xgboost_engine.evaluate_parts(
                    self._evaluation_snapshot, report['selected_part_results'],
                    self._evaluation_options, direction=winning_dir)
                if not isinstance(self.last_xgboost_report, dict):
                    raise ValueError('XGBoost report must be a dictionary')
            except Exception as error:
                self.last_xgboost_report = {'available': False, 'approved': False,
                    'reason': 'inference_error', 'error': f'{type(error).__name__}: {error}'}
            if type(self.last_xgboost_report.get('available')) is not bool or type(self.last_xgboost_report.get('approved')) is not bool:
                self.last_xgboost_report = {'available': False, 'approved': False, 'reason': 'invalid_model_report'}
            if not self.last_xgboost_report.get('available'):
                if self.require_xgboost:
                    return extended_hold('XGBOOST_UNAVAILABLE: ' + str(self.last_xgboost_report.get('reason', 'unknown')))
                else:
                    self.last_xgboost_report = dict(self.last_xgboost_report)
                    self.last_xgboost_report['neutral_fallback'] = True
                    self.last_xgboost_report['probability'] = 0.5
                    self.last_xgboost_report['veto'] = False
                    _fm = (self.last_xgboost_report.get('feature_report') or {}).get('missing_features')
                    if _fm is not None:
                        self.last_xgboost_report['populated_count'] = '%d/48' % (48 - len(_fm))
                        self.last_xgboost_report['missing_features_list'] = list(_fm)
            elif not self.last_xgboost_report.get('approved'):
                return extended_hold('XGBOOST_PARTS_VETO')

        # XGBoost veto note: the previous live_godmode.xgb was trained on
        # random vote features. The active XGBoost layer is now the
        # validated, per-symbol 15m meta-gate in jarvis_xgb_meta.py, applied in
        # jarvis_FIXED._xgb_meta_gate (veto-only, fail-neutral).

        # --- END XGBOOST VETO ---

        if not winning_dir:
            return {
                "action": "HOLD",
                "trade_type": None,
                "direction": None,
                "entry": price,
                "sl": None,
                "tp": None,
                "rr": None,
                "total_brains_voted": len(votes),
                "timeframes_evaluated": list(tf_dict.keys()),
                "all_brain_votes": votes,
                "vote_tally": tally,
                "reason": f"NO_{len(votes)}_BRAIN_CONSENSUS",
                "xgb_shadow": getattr(self, "last_xgb_report", {})
            }

        sl, tp, rr = self.calculate_dynamic_sl_tp(m5 if winning_mode == "SCALP" else h1, price, winning_dir, winning_mode)

        result = {
            "action": "EXECUTE",
            "trade_type": winning_mode,
            "direction": winning_dir,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "total_brains_voted": len(votes),
            "timeframes_evaluated": list(tf_dict.keys()),
            "all_brain_votes": votes,
            "vote_tally": tally,
            "reason": f"ALL_{len(votes)}_BRAIN_CONSENSUS_{winning_mode}_{winning_dir}",
            "xgb_shadow": getattr(self, "last_xgb_report", {})
        }

        # Let the internal brains know the final verdict via the Cognitive Bus
        if hasattr(self, 'p2_system') and hasattr(self.p2_system, 'bus'):
            try:
                self.p2_system.bus.publish('SYSTEM', 'ModeEngine', 
                    f"Consensus Reached: {winning_mode} {winning_dir} (Score: {tally[(winning_mode, winning_dir)]}). Informing all internal brains.")
            except Exception:
                pass
                
        return result
