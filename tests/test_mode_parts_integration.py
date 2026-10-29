"""Offline Parts 4–12 wiring tests; no network, GPU or order execution."""
import ast
import copy
import importlib.util
import logging
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
from mode_engine import PartEngineBridge, PART_CLASSES, VOTE_NAMES, direction


def frame():
    price = 100.0
    rows = []
    for i in range(100):
        old = price
        price *= math.exp(.0005 + .0001 * math.sin(i * .4))
        rows.append({'open': old, 'high': price * 1.001, 'low': old * .999,
                     'close': price, 'volume': 100 + i % 7})
    return pd.DataFrame(rows)


class Engine:
    def __init__(self, name, signal=1, confidence=80):
        self.name, self.signal, self.confidence, self.calls = name, signal, confidence, []
    def analyze(self, data, **kwargs):
        self.calls.append((data, kwargs))
        return {'signal': self.signal, 'confidence': self.confidence, 'thought': 'healthy', 'telemetry': {}}


def engines():
    return {key: Engine(key) for key in PART_CLASSES}


def seed():
    return {key: {'signal': 1, 'thought': 'native'} for key in
            ('part1_breakout', 'part2_zone', 'part3_psychology')}


class BridgeTests(unittest.TestCase):
    def test_loads_all_nine_requested_classes(self):
        seen = []
        def importer(module):
            name, cls = next((key, pair[1]) for key, pair in PART_CLASSES.items() if pair[0] == module)
            seen.append((module, cls))
            return types.SimpleNamespace(**{cls: lambda: Engine(name)})
        bridge = PartEngineBridge(importer=importer)
        self.assertEqual(len(seen), 9)
        self.assertEqual(set(bridge.engines), set(PART_CLASSES))

    def test_ohlcv_for_4_to_10_and_named_results_for_11_12(self):
        native = engines(); bridge = PartEngineBridge(native)
        data = frame()
        for tf in ('1m', '5m', '1h'):
            bridge.evaluate_frame(tf, data, seed(), {'symbol': 'BTCUSDT'})
        for key in VOTE_NAMES:
            self.assertEqual(len(native[key].calls), 3)
            self.assertIsInstance(native[key].calls[0][0], pd.DataFrame)
            self.assertEqual(native[key].calls[0][1]['context']['timeframe'], '1m')
        for key in ('part11_fusion', 'part12_confidence'):
            inputs, kwargs = native[key].calls[0]
            self.assertEqual(set(inputs), set(seed()) | set(VOTE_NAMES))
            self.assertEqual(kwargs, {})
            self.assertNotIn('part11_fusion', inputs)
        self.assertIsNone(bridge.gate('SWING', 'BUY'))
        self.assertEqual(bridge.report('SWING')['selected_timeframe'], '1h')

    def test_analyzers_cannot_mutate_shared_candles_or_named_inputs(self):
        native = engines(); data = frame(); original = data.copy(deep=True)
        def mutate_frame(payload, **kwargs):
            payload.loc[:, 'close'] = 0
            return {'signal': 1, 'confidence': 80}
        def mutate_inputs(payload):
            payload['part1_breakout']['signal'] = -1
            return {'signal': 1, 'confidence': 80}
        native['part4_volume'].analyze = mutate_frame
        native['part11_fusion'].analyze = mutate_inputs
        bridge = PartEngineBridge(native)
        output = bridge.evaluate_frame('5m', data, seed())
        pd.testing.assert_frame_equal(data, original)
        self.assertEqual(output['part1_breakout']['signal'], 1)
        self.assertEqual(native['part12_confidence'].calls[0][0]['part1_breakout']['signal'], 1)

    def test_native_exception_is_visible_and_math_fallback_is_real(self):
        native = engines()
        native['part6_trend'].analyze = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('native failure'))
        bridge = PartEngineBridge(native)
        out = bridge.evaluate_frame('5m', frame(), seed())
        self.assertEqual(out['part6_trend']['runtime_status'], 'math_fallback')
        self.assertIn('native failure', out['part6_trend']['native_error'])
        self.assertIn('part6_trend', bridge.errors['5m'])
        self.assertIn(out['part6_trend']['signal'], (-1, 0, 1))

    def test_fusion_unavailable_or_neutral_is_not_a_free_pass(self):
        native = engines(); native.pop('part11_fusion')
        bridge = PartEngineBridge(native)
        bridge.evaluate_frame('5m', frame(), seed())
        self.assertEqual(bridge.gate('SCALP', 'BUY'), 'PART11_FUSION_UNAVAILABLE')
        native = engines(); native['part11_fusion'].signal = 0
        bridge = PartEngineBridge(native); bridge.evaluate_frame('5m', frame(), seed())
        self.assertEqual(bridge.gate('SCALP', 'BUY'), 'PART11_FUSION_NO_CONFIRMATION')

    def test_part12_low_nan_confidence_blocks_without_direction_vote(self):
        native = engines(); native['part12_confidence'].confidence = 40
        bridge = PartEngineBridge(native); bridge.evaluate_frame('5m', frame(), seed())
        self.assertEqual(bridge.gate('SCALP', 'BUY'), 'PART12_LOW_CONFIDENCE')
        native['part12_confidence'].confidence = float('nan')
        bridge.reset(); bridge.evaluate_frame('5m', frame(), seed())
        self.assertEqual(bridge.gate('SCALP', 'BUY'), 'PART12_CONFIDENCE_UNAVAILABLE')
        self.assertNotIn('part12_confidence', VOTE_NAMES)

    def test_part7_risk_advisory_on_higher_tf_and_blocks_on_entry_tf(self):
        native = engines(); bridge = PartEngineBridge(native)
        bridge.evaluate_frame('5m', frame(), seed())
        native['part7_volatility'].analyze = lambda *a, **k: {'signal': 0, 'confidence': 5, 'telemetry': {'risk_veto': True}}
        bridge.evaluate_frame('1h', frame(), seed())
        self.assertIsNone(bridge.gate('SCALP', 'BUY'))
        bridge.evaluate_frame('5m', frame(), seed())
        self.assertEqual(bridge.gate('SCALP', 'BUY'), 'PART7_VOLATILITY_VETO')

    def test_real_current_analyzer_bodies_run_without_heavy_imports(self):
        native = {}
        for key in VOTE_NAMES:
            module, name = PART_CLASSES[key]
            tree = ast.parse((ROOT / (module + '.py')).read_text(encoding='utf-8'))
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
            method = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'analyze'))
            method.decorator_list = []
            wrapper = ast.ClassDef(name='Harness', bases=[], keywords=[], body=[method], decorator_list=[])
            future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
            code = ast.fix_missing_locations(ast.Module(body=[future, wrapper], type_ignores=[]))
            ns = {'pd': pd, 'np': np}; exec(compile(code, module, 'exec'), ns)
            native[key] = ns['Harness']()
        # Execute the actual Part 11/12 classes with an unavailable torch shim.
        torch = types.SimpleNamespace(device=lambda x: x, cuda=types.SimpleNamespace(is_available=lambda: False))
        for key in ('part11_fusion', 'part12_confidence'):
            module, name = PART_CLASSES[key]
            tree = ast.parse((ROOT / (module + '.py')).read_text(encoding='utf-8'))
            cls = copy.deepcopy(next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name))
            cls.body = [node for node in cls.body if not isinstance(node, ast.FunctionDef) or node.name != 'analyze_multi_timeframe']
            code = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[]))
            ns = {'torch': torch, 'TORCH_AVAILABLE': False, 'math': math, 'logging': logging, 'xgb_shadow': lambda n, m: lambda c: c}
            exec(compile(code, module, 'exec'), ns)
            native[key] = ns[name]()
        bridge = PartEngineBridge(native)
        out = bridge.evaluate_frame('5m', frame(), seed())
        for key in PART_CLASSES:
            self.assertEqual(out[key]['runtime_status'], 'native', (key, out[key]))
            self.assertNotIn('error:', out[key].get('thought', '').lower())
            self.assertNotIn('insufficient', out[key].get('thought', '').lower())
        self.assertEqual(len(out), 12)


def native_part7(numpy_module=np):
    tree = ast.parse((ROOT / 'part7_FIXED.py').read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'VolatilityEngineGPU')
    method = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'analyze'))
    method.decorator_list = []
    wrapper = ast.ClassDef(name='Harness', bases=[], keywords=[], body=[method], decorator_list=[])
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    ns = {'pd': pd, 'np': numpy_module}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, wrapper], type_ignores=[])), 'native_part7', 'exec'), ns)
    return ns['Harness']()


class P1Brain:
    def __getattr__(self, name):
        outputs = {
            'analyze_trend': {'support_score': .8, 'trend_direction': 1},
            'analyze_volatility': {'volatility_regime': 'HIGH'},
            'analyze_strength': {'overall_strength': .8},
            'analyze_risk': {'risk_level': 'LOW', 'stop_distance': .5},
            'analyze_reversal': {'reversal_probability': 0},
            'analyze_regime': {'regime_type': 'BULL', 'regime_confidence': .8},
            'analyze_deepseek': {'fakeout_block': .1},
            'analyze_evolution': {'support_score': .8}, 'analyze_memory': {'support_score': .8},
            'analyze_self_heal': {'health_index': .8}, 'analyze_mini_r1': {'trap_probability': .1},
            'analyze_mini_v3': {'tick_momentum': .8}, 'analyze_meta_fusion': {'support_score': .8},
        }
        return lambda *args, **kwargs: dict(outputs[name])


class P2Brain:
    def __getattr__(self, name):
        def call(*args, **kwargs):
            if name == 'analyze_candle_psychology': return {'is_bullish': True, 'is_bearish': False}
            if name == 'get_nearest_levels': return {'support': 99, 'resistance': 102}
            if name.startswith(('get_', 'detect_0_5', 'detect_volume_signals', 'fuse_')): return [('BUY', 7)]
            return {'alignment': 'BUY'}
        return call


def mode_class():
    """Import production ModeEngine while replacing heavy Part imports only."""
    p1 = types.ModuleType('part1_FIXED')
    for cls in ('TrendBrain', 'VolatilityBrain', 'StrengthBrain', 'RiskBrain', 'ReversalBrain', 'RegimeBrain',
                'DeepSeekBrain', 'EvolutionBrain', 'MemoryBrain', 'SelfHealingBrain', 'MetaFusionBrain', 'MiniR1Brain', 'MiniV3Brain'):
        setattr(p1, cls, P1Brain)
    p2 = types.ModuleType('part2_FIXED')
    names = ('candle_psychology', 'zone_detector', 'volume_profile', 'market_structure', 'order_flow',
             'momentum_oscillator', 'volatility_regime', 'cycle_analysis', 'correlation_matrix', 'pattern_recognition',
             'support_resistance', 'trend_analysis', 'market_regime', 'price_action', 'institutional_flow', 'signal_fusion')
    p2.AdvancedAnalysisSystem = lambda: types.SimpleNamespace(brains={name: P2Brain() for name in names})
    p3 = types.ModuleType('part3_FIXED')
    p3.InstitutionalTradingEngineGPU = lambda: types.SimpleNamespace(generate_mtf_signals=lambda data: {'mtf_consensus': .8, 'components': {'regime': 'TRENDING'}})
    p5 = types.ModuleType('part5_FIXED')
    p5.InstitutionalFusionEngine = lambda: types.SimpleNamespace(process_market_data=lambda *args: {'fused_signal': 'NO TRADE'})
    bus = types.ModuleType('jarvis_cognitive_bus'); bus.CognitiveBus = lambda: None
    spec = importlib.util.spec_from_file_location('_mode_engine_under_test', ROOT / 'mode_engine.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'part1_FIXED': p1, 'part2_FIXED': p2, 'part3_FIXED': p3, 'part5_FIXED': p5, 'jarvis_cognitive_bus': bus}):
        spec.loader.exec_module(module)
        # Constructor has one dynamic p5 import, so callers construct inside patch.
        yield module.ModeEngine


class XGB:
    def __init__(self, available=True, approved=True):
        self.available, self.approved, self.calls = available, approved, []
    def evaluate_parts(self, snapshot, results, options, direction):
        self.calls.append((snapshot, results, options, direction))
        return {'available': self.available, 'approved': self.approved, 'reason': 'test_model'}


class ModeIntegrationTests(unittest.TestCase):
    def evaluate(self, xgb=None, native=None, required=True):
        for cls in mode_class():
            bridge = PartEngineBridge(native or engines())
            engine = cls(part_bridge=bridge, xgboost_engine=xgb or XGB(), require_xgboost=required)
            data = frame()
            result = engine.master_evaluate(data, data, {'5m': data, '1h': data}, snapshot={'symbol': 'BTCUSDT'}, options_data={'pcr': 1})
            return engine, result

    def test_new_votes_and_all_parts_reach_model(self):
        xgb = XGB(); engine, out = self.evaluate(xgb=xgb)
        self.assertEqual(out['action'], 'EXECUTE')
        self.assertEqual(out['total_brains_voted'], 37)  # 13 P1 + 16 P2 + 1 P3 + 7 Parts(4-10)
        self.assertEqual(out['consensus_thresholds'], {'SWING': 7, 'SCALP': 8})
        for name in VOTE_NAMES.values(): self.assertIn(name, out['all_brain_votes'])
        self.assertFalse(any(name.startswith(('P11_', 'P12_')) for name in out['all_brain_votes']))
        self.assertEqual(set(xgb.calls[0][1]), set(seed()) | set(PART_CLASSES))
        self.assertEqual(xgb.calls[0][3], 'BUY')
        self.assertEqual(out['part_engine_report']['selected_timeframe'], '1h')
        self.assertEqual(out['part12_confidence'], 80)

    def test_unavailable_model_holds_by_default(self):
        _, out = self.evaluate(xgb=XGB(available=False))
        self.assertEqual(out['action'], 'HOLD')
        self.assertTrue(out['reason'].startswith('XGBOOST_UNAVAILABLE'))
        self.assertFalse(out['xgboost_report']['available'])

    def test_actual_xgboost_engine_receives_bridge_results_and_reports_gaps(self):
        from jarvis_xgboost_engine import JarvisXGBoostEngine
        model = JarvisXGBoostEngine(model_path='/no/such/model.json', allow_bootstrap=False)
        _, out = self.evaluate(xgb=model)
        self.assertEqual(out['action'], 'HOLD')
        self.assertEqual(out['xgboost_report']['reason'], 'missing_required_features')
        report = out['xgboost_report']['feature_report']
        self.assertEqual(len(report['parts_provided']), 12)
        self.assertIn('part12_confidence', report['parts_provided'])
        self.assertIn('p12_historical_accuracy_weight', report['missing_features'])
        self.assertNotIn('part12_confidence', report['parts_used'])
        self.assertEqual(out['part12_confidence'], 80)

    def test_explicit_algorithm_only_mode_is_reported(self):
        _, out = self.evaluate(xgb=XGB(available=False), required=False)
        self.assertEqual(out['action'], 'EXECUTE')
        self.assertFalse(out['xgboost_report']['available'])

    def test_available_model_rejection_holds_even_optional(self):
        _, out = self.evaluate(xgb=XGB(approved=False), required=False)
        self.assertEqual(out['reason'], 'XGBOOST_PARTS_VETO')

    def test_part7_gate_is_not_bypassed_by_part5_override(self):
        native = engines()
        native['part7_volatility'].analyze = lambda *a, **k: {'signal': 0, 'confidence': 5, 'telemetry': {'entry_blocked': True}}
        xgb = XGB(); engine, out = self.evaluate(xgb=xgb, native=native)
        engine.p5_fusion = types.SimpleNamespace(process_market_data=lambda *a: {'fused_signal': 'BUY'})
        data = frame()
        out = engine.master_evaluate(data, data, {'1m': data, '5m': data, '1h': data})
        self.assertEqual(out['reason'], 'PART7_VOLATILITY_VETO')
        self.assertEqual(out['vote_tally'][('SWING', 'BUY')], 100)
        self.assertEqual(len(xgb.calls), 0)

    def test_real_part7_atr_hmm_and_context_vetoes(self):
        for scenario in ('atr', 'hmm', 'context'):
            with self.subTest(scenario=scenario):
                native = engines(); native['part7_volatility'] = native_part7()
                for cls in mode_class():
                    xgb = XGB(); engine = cls(part_bridge=PartEngineBridge(native), xgboost_engine=xgb)
                    data = frame(); context = {}
                    if scenario == 'atr':
                        data['high'] = data[['open', 'close']].max(axis=1) * 1.02
                        data['low'] = data[['open', 'close']].min(axis=1) * .98
                    elif scenario == 'hmm':
                        data['open'] = 100
                        data['close'] = [100 * (1.03 if i % 2 else .97) for i in range(len(data))]
                        data['high'] = data[['open', 'close']].max(axis=1) * 1.001
                        data['low'] = data[['open', 'close']].min(axis=1) * .999
                    else:
                        context = {'institutional_components': {'regime': 'PANIC'}}
                    out = engine.master_evaluate(data, data, {'5m': data, '1h': data}, snapshot=context)
                    self.assertEqual(out['reason'], 'PART7_VOLATILITY_VETO')
                    self.assertEqual(out['action'], 'HOLD')
                    self.assertEqual(len(xgb.calls), 0)
                    raw = out['part_engine_report']['selected_part_results']['part7_volatility']
                    self.assertTrue(raw['risk_veto'])

    def test_native_part3_panic_reaches_part7_and_cannot_be_erased(self):
        native = engines(); native['part7_volatility'] = native_part7()
        for cls in mode_class():
            xgb = XGB(); engine = cls(part_bridge=PartEngineBridge(native), xgboost_engine=xgb)
            engine.p3_institutional = types.SimpleNamespace(generate_mtf_signals=lambda data: {
                'mtf_consensus': .8, 'components': {'regime': 'PANIC'}})
            data = frame()
            out = engine.master_evaluate(data, data, {'5m': data, '1h': data},
                                         snapshot={'institutional_components': {'regime': 'NEUTRAL'}})
            self.assertEqual(out['action'], 'HOLD')
            self.assertEqual(out['reason'], 'PART7_VOLATILITY_VETO')
            self.assertEqual(len(xgb.calls), 0)

    def test_real_part7_short_frames_and_internal_errors_block(self):
        for bad_numpy, short in ((False, True), (True, False)):
            native = engines(); native['part7_volatility'] = native_part7(None if bad_numpy else np)
            for cls in mode_class():
                xgb = XGB(); engine = cls(part_bridge=PartEngineBridge(native), xgboost_engine=xgb)
                data = frame().iloc[:10] if short else frame()
                out = engine.master_evaluate(data, data, {'5m': data, '1h': data})
                self.assertEqual(out['action'], 'HOLD')
                self.assertEqual(out['reason'], 'PART7_VOLATILITY_UNAVAILABLE')
                self.assertEqual(len(xgb.calls), 0)

    def test_invalid_and_missing_inputs_return_structured_hold(self):
        cases = [frame().drop(columns='volume'), frame().assign(close=float('nan')),
                 frame().assign(close=0), frame().assign(volume=-1)]
        for data in cases:
            for cls in mode_class():
                engine = cls(part_bridge=PartEngineBridge(engines()), xgboost_engine=XGB())
                out = engine.master_evaluate(data, data, {'5m': data})
                self.assertEqual(out['action'], 'HOLD')
                self.assertTrue(out['reason'].startswith('INVALID_MARKET_DATA'))
        for cls in mode_class():
            engine = cls(part_bridge=PartEngineBridge(engines()), xgboost_engine=XGB())
            data = frame()
            out = engine.master_evaluate(None, data, {'1h': data})
            self.assertEqual(out['action'], 'EXECUTE')
            self.assertGreater(out['entry'], 0)
            out = engine.master_evaluate(data, None, {'5m': data})
            self.assertEqual(out['action'], 'EXECUTE')
            self.assertEqual(out['trade_type'], 'SCALP')
            engine.p5_fusion = types.SimpleNamespace(process_market_data=lambda *a: {'fused_signal': 'BUY'})
            out = engine.master_evaluate(data, None, {'1m': data, '5m': data})
            self.assertEqual(out['action'], 'HOLD')
            self.assertEqual(out['reason'], 'MISSING_EXECUTION_TIMEFRAME')
            out = engine.master_evaluate(None, None, {})
            self.assertTrue(out['reason'].startswith('INVALID_MARKET_DATA'))

    def test_malformed_numeric_model_flags_cannot_authorize(self):
        xgb = XGB()
        xgb.evaluate_parts = lambda *a, **k: {'available': 1, 'approved': 1}
        _, out = self.evaluate(xgb=xgb)
        self.assertEqual(out['action'], 'HOLD')
        self.assertEqual(out['xgboost_report']['reason'], 'invalid_model_report')

    def test_legacy_opt_out_keeps_29_votes(self):
        for cls in mode_class():
            engine = cls(enable_extended_parts=False)
            out = engine.evaluate_all_brains({'5m': frame()})
            self.assertEqual(len(out), 29)
            self.assertEqual(engine.SWING_CONSENSUS_THRESHOLD, 5)


if __name__ == '__main__': unittest.main()
