from central_entry_fixture import entry_evidence
"""Offline production-module behavior checks; no network, model inference or orders."""
import os
import unittest
from unittest.mock import Mock, patch

from ai_hedge_advisor import AIHedgeAdvisor
from jarvis_neural_cortex import JarvisNeuralCortex
from jarvis_market_oracle import JarvisMarketOracle
from jarvis_specialist_pool import SpecialistPool
from multi_ai_consensus import run_ai_roundtable
from options_hedged_scalp import OptionsHedgedScalpEngine


class MigrationIntegration(unittest.TestCase):
    def setUp(self):
        self.block_post = patch('requests.post', side_effect=AssertionError('model/network post'))
        self.block_get = patch('requests.get', side_effect=AssertionError('model/network get'))
        self.block_post.start(); self.block_get.start()
        self.addCleanup(self.block_post.stop); self.addCleanup(self.block_get.stop)

    def test_cortex_math_not_model_and_no_cached_vote(self):
        cortex = JarvisNeuralCortex()
        cortex._call_legacy_llm_chat = Mock(side_effect=AssertionError('model called'))
        result = cortex.analyze({'part11_fusion': {'signal': -1},
                                 'part12_confidence': {'confidence': 66}}, 101.0)
        self.assertEqual(result['signal'], 'PUT')
        self.assertFalse(result['ai_online'])
        cortex._call_legacy_llm_chat.assert_not_called()
        self.assertEqual(cortex.analyze_holistic_context('prompt')[0], '')

    def test_hedge_and_paper_execution_ignore_model_even_when_env_disabled(self):
        with patch.dict(os.environ, {'JARVIS_PURE_ALGO': 'false'}):
            advisor = AIHedgeAdvisor()
            self.assertEqual(advisor.evaluate_hedge_setup('BUY', 92, 100, 2, {}, 1)['hedge'], 'NO')
            self.assertEqual(advisor.monitor_active_hedge({}, 100, -1, 1)['action'], 'HOLD')
            forbidden = Mock()
            forbidden.evaluate_hedge_setup.side_effect = AssertionError('hedge model called')
            paper = OptionsHedgedScalpEngine(None, forbidden, None)
            paper.start_monitor = Mock()  # No background price feed or side effects in tests.
            out = paper.execute_hedged_scalp({'direction': 'BUY', 'confidence': 80},
                                              100., 2., {'calls': [], 'puts': []})
            self.assertEqual(out['status'], 'executed')
            self.assertFalse(out['hedge_applied'])
            self.assertIsNone(paper.active_positions[out['position_id']]['hedge_leg'])
            paper.start_monitor.assert_called_once()
            forbidden.evaluate_hedge_setup.assert_not_called()

    def test_oracle_never_fabricates_missing_data_or_calls_model(self):
        oracle = JarvisMarketOracle.__new__(JarvisMarketOracle)
        oracle._gemini_client = Mock()
        no_data = {'btc_spot': 0, 'deribit': {}, 'microstructure': {}}
        board = oracle._run_ai_board(no_data)
        self.assertEqual(board['verdict'], 'WAIT')
        for method in (oracle._call_ai_oracle, oracle._call_gemini_oracle):
            self.assertEqual(method(no_data, board)['trade_suggestion'], 'WAIT')
        oracle._gemini_client.models.generate_content.assert_not_called()
        observed = {'btc_spot': 100., 'deribit': {'pcr': .70, 'max_pain': 105.},
                    'microstructure': {'imbalance_pct': 2.}}
        fc = oracle._call_ai_oracle(observed, board)
        self.assertEqual(fc['model_used'], 'local_synthesizer')
        self.assertIn(fc['trade_suggestion'], ('CALL', 'PUT', 'WAIT'))
        self.assertEqual(oracle._call_ai_oracle({'btc_spot': 100., 'deribit': {},
                                                     'microstructure': {}}, board)['trade_suggestion'], 'WAIT')

    def test_oracle_gate_rejects_model_and_stale_maps(self):
        from datetime import datetime, timedelta, timezone
        from oracle_trade_gate import OracleTradeGate
        oracle = Mock()
        now = datetime.now(timezone.utc)
        forecast = {'model_used': 'gemini-3.6-flash', 'generated_at': now.isoformat(),
                    '5min': {'direction': 'BULLISH'}, '30min': {'direction': 'BULLISH'},
                    'trade_suggestion': 'CALL', 'entry_zone': {'price_from': 99, 'price_to': 101}}
        oracle.get_latest_forecast.return_value = forecast
        gate = OracleTradeGate(oracle_ref=oracle, hard_gate=False)
        self.assertFalse(gate.is_trade_aligned('CALL')[0])
        self.assertFalse(gate.is_price_in_entry_zone(100)[0])
        forecast['model_used'] = 'local_synthesizer'
        forecast['generated_at'] = (now - timedelta(minutes=12)).isoformat()
        self.assertFalse(gate.is_trade_aligned('CALL')[0])
        forecast['generated_at'] = now.isoformat()
        self.assertTrue(gate.is_trade_aligned('CALL')[0])
        self.assertTrue(gate.is_price_in_entry_zone(100)[0])

    def test_auto_trader_model_hooks_cannot_override_risk_or_create_hedge(self):
        from jarvis_live_trader import JarvisAutoTrader
        from tests.test_execution_safety import DeltaStub
        forbidden = Mock()
        forbidden.evaluate_hedge_setup.side_effect = AssertionError('model-selected hedge')
        forbidden.get_confidence_threshold.side_effect = AssertionError('model risk threshold')
        forbidden.is_trading_allowed.side_effect = AssertionError('model entry veto')
        delta = DeltaStub(100)
        trader = JarvisAutoTrader(delta, hedge_advisor=forbidden)
        trader.is_enabled = False
        trader._gemini_advisor = forbidden
        self.assertFalse(trader._get_hedge_plan('CALL', 90, 100., {}, 'BTCUSDT')['do_hedge'])
        self.assertTrue(trader._check_risk_gates(90, gemini_advisor=forbidden)['ok'])
        forbidden.evaluate_hedge_setup.assert_not_called()
        forbidden.get_confidence_threshold.assert_not_called()
        with patch('jarvis_live_trader._get_oracle_gate', return_value=None), \
             patch('jarvis_live_trader._get_sizer', return_value=None):
            # Direct private caller cannot slip a model-selected leg into even
            # a paper position. No order or exchange mutation is permitted.
            result = trader._place_trade('CALL', 90, 100., entry_evidence('BTCUSDT', 'CALL', 90, 100.)[1], {'do_hedge': True, 'strike': 105, 'option_type': 'PUT'}, 'BTCUSDT', **entry_evidence('BTCUSDT', 'CALL', 90, 100.)[0])
        self.assertTrue(result['success'])
        self.assertIsNone(result['hedge'])
        self.assertEqual(delta.order_calls, [])
        self.assertEqual(delta.leverage_calls, [])
        forbidden.is_trading_allowed.assert_not_called()

    def test_oracle_exit_never_applies_btc_forecast_to_alt_symbol(self):
        from datetime import datetime, timezone
        from oracle_trade_gate import OracleTradeGate
        oracle = Mock()
        oracle.get_latest_forecast.return_value = {
            'model_used': 'local_synthesizer', 'generated_at': datetime.now(timezone.utc).isoformat(),
            'exit_target': 101., 'stop_loss': 99., 'hold_minutes': 15}
        gate = OracleTradeGate(oracle_ref=oracle)
        self.assertFalse(gate.get_oracle_tp_sl(100., 'CALL', symbol='ETHUSDT')['use_oracle'])
        self.assertTrue(gate.get_oracle_tp_sl(100., 'CALL', symbol='BTCUSDT')['use_oracle'])

    def test_retired_committee_cannot_confirm(self):
        pool = SpecialistPool()
        opinions = pool.run_parallel_evaluation({'symbol': 'BTCUSDT'})
        result = pool.synthesize_chairman_decision({}, opinions)
        self.assertFalse(result['approved'])
        self.assertEqual(result['approve_votes'], 0)
        self.assertFalse(run_ai_roundtable({}, {})['approved'])


if __name__ == '__main__': unittest.main()
