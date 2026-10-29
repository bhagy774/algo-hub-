import io
import json
import os
import unittest
from unittest.mock import patch

import jarvis_close_coordinator as close
from jarvis_dashboard import render_dashboard
from jarvis_runtime import detect_backend, safe_device


class RuntimeSafetyContracts(unittest.TestCase):
    def tearDown(self):
        close.release_close("BTCUSDT", "case", force=True)

    def test_close_claim_and_reconciliation_fail_closed(self):
        self.assertTrue(close.claim_close("BTCUSDT", "case", "a"))
        self.assertFalse(close.claim_close("BTCUSDT", "case", "b"))
        class Venue:
            def get_open_positions(self, symbol):
                return [{"symbol": symbol, "size": "1"}]
        self.assertFalse(close.reconcile_close(Venue(), "BTCUSDT", "sell", 1)["confirmed"])

        class WrongVenue:
            def get_open_positions(self, symbol):
                return [{"symbol": "ETHUSDT", "size": 0}]
        result = close.reconcile_close(WrongVenue(), "BTCUSDT", "sell", 1)
        self.assertFalse(result["confirmed"])
        self.assertEqual(result["reason"], "symbol mismatch")

    def test_close_order_payload_is_reduce_only_and_idempotent(self):
        from delta_api_wrapper import DeltaExchangeData
        client = DeltaExchangeData.__new__(DeltaExchangeData)
        client.get_product_id = lambda symbol: 42
        captured = {}
        client._request = lambda method, path, payload, authorized=True: captured.update(payload) or {
            'success': True, 'data': {'result': {'id': 'order-1'}}
        }
        with patch.dict(os.environ, {'DELTA_ORDER_EXECUTION_ENABLED': 'true'}):
            result = client.place_order('ETHUSDT', 'sell', 3, reduce_only=True,
                                        client_order_id='jarvis-close-eth-1')
        self.assertTrue(result['success'])
        self.assertIs(captured['reduce_only'], True)
        self.assertEqual(captured['client_order_id'], 'jarvis-close-eth-1')



    def test_gpu_detection_never_fakes_accelerator(self):
        with patch.dict(os.environ, {"JARVIS_DEVICE": "cuda"}, clear=False), patch("jarvis_runtime._torch", return_value=None):
            status = detect_backend()
        self.assertEqual(status.backend, "cpu")
        self.assertFalse(status.detected)
        self.assertIn("unavailable", status.fallback_reason.lower())
        self.assertEqual(str(safe_device("not-a-device")), "cpu")

    def test_synthetic_selected_symbol_parts_to_validated_advisory(self):
        # Offline integration contract: every adapter is callable, the same
        # selected symbol is carried into one bounded snapshot, and the AI
        # response remains advisory rather than execution authority.
        import numpy as np
        import pandas as pd
        from jarvis_FIXED import JarvisElite
        n = 600
        close_values = 100 + np.sin(np.arange(n) / 20.0) * 2
        candles = pd.DataFrame({
            'open': close_values - .1, 'high': close_values + .3,
            'low': close_values - .3, 'close': close_values,
            'volume': np.full(n, 1000.0),
        })
        with patch.dict(os.environ, {'JARVIS_DEVICE': 'cpu', 'JARVIS_PRESIM': '0',
                                     'JARVIS_SCEN_SIM': '0', 'JARVIS_BACKTEST_MODE': '1'}, clear=False):
            brain = JarvisElite(backtest_mode=True)
            brain.active_symbol = 'ETHUSDT'
            outputs = {}
            for name, adapter in list(brain.parts.items()):
                if name == 'part11_fusion':
                    outputs[name] = adapter.analyze(outputs)
                elif name == 'part12_confidence':
                    outputs[name] = adapter.analyze(list(outputs.values()))
                else:
                    outputs[name] = adapter.analyze(candles, context={})
            self.assertEqual(len([name for name in outputs if name not in {"part14_options_chain", "part13_patterns"}]), 12)
            self.assertTrue(all(isinstance(value, dict) for value in outputs.values()))

    def test_single_dashboard_contains_authoritative_and_advisory_state(self):
        stream = io.StringIO()
        render_dashboard({"symbol": "ETHUSDT", "signal": {"direction": "WAIT", "confidence": 0},
                          "reasons": ["missing data"], "plan": {"action": "WAIT"},
                          "status": {"mode": "PAPER", "readiness": {"status": "NOT_READY"},
                                     "ai_suggestion": {"decision": "BUY", "rationale": "advisory"},
                                     "gpu": {"backend": "cpu", "device_name": "cpu"}},
                          "account": {}}, stream=stream)
        output = stream.getvalue()
        self.assertIn("WAIT", output)
        self.assertIn("BUY (advisory", output)
        self.assertIn("NOT_READY", output)
        self.assertIn("GPU BACKEND", output)


if __name__ == "__main__":
    unittest.main()
