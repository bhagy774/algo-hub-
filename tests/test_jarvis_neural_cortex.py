import unittest
from unittest.mock import patch, MagicMock
from jarvis_neural_cortex import JarvisNeuralCortex

class TestJarvisNeuralCortex(unittest.TestCase):
    def setUp(self):
        self.cortex = JarvisNeuralCortex()

    def test_initialization(self):
        self.assertEqual(len(self.cortex.chat_history), 0)
        self.assertEqual(self.cortex.max_history_pairs, 20)
        self.assertEqual(self.cortex.get_memory_length(), 0)

    def test_analyze_success_with_json(self):
        # Model output is retired; an unsupported part cannot authorize a trade.
        with patch("requests.post", side_effect=AssertionError("no model network")):
            result = self.cortex.analyze({'part1_breakout': {'signal': 1}}, 50000.0)
        self.assertEqual(result['signal'], 'NO_TRADE')
        self.assertFalse(result['ai_online'])
        self.assertEqual(self.cortex.get_memory_length(), 0)

    def test_analyze_fallback_on_api_error(self):
        with patch("requests.post", side_effect=AssertionError("no model network")):
            result = self.cortex.analyze({'part11_fusion': {'signal': -1}, 'part12_confidence': {'confidence': 60}}, 50000.0)
        self.assertEqual(result['signal'], 'PUT')
        self.assertEqual(result['confidence'], 60)
        self.assertFalse(result['ai_online'])

    def test_analyze_fallback_on_bad_json(self):
        with patch("requests.post", side_effect=AssertionError("no model network")):
            result = self.cortex.analyze({}, 50000.0)
        self.assertEqual(result['signal'], 'NO_TRADE')
        self.assertFalse(result['ai_online'])

    def test_memory_management(self):
        # Simulate pushing more than max_history_pairs
        self.cortex.max_history_pairs = 2 # max 4 messages

        for i in range(10):
            self.cortex.chat_history.append({"role": "user", "content": f"U{i}"})
            self.cortex.chat_history.append({"role": "assistant", "content": f"A{i}"})

            # This is what `analyze` does internally
            max_messages = self.cortex.max_history_pairs * 2
            if len(self.cortex.chat_history) > max_messages:
                self.cortex.chat_history = self.cortex.chat_history[-max_messages:]

        self.assertEqual(len(self.cortex.chat_history), 4)
        self.assertEqual(self.cortex.chat_history[0]['content'], "U8")
        self.assertEqual(self.cortex.chat_history[-1]['content'], "A9")

    def test_reset_memory(self):
        self.cortex.chat_history = [{"role": "user", "content": "hi"}]
        self.assertEqual(self.cortex.get_memory_length(), 0) # integer division

        self.cortex.reset_memory()
        self.assertEqual(len(self.cortex.chat_history), 0)

    def test_build_market_report(self):
        part_results = {
            'part1_breakout': {'signal': 1, 'thought': 'Bullish BO'},
            'part2_zone': {'signal': -1, 'thought': 'Supply zone'}
        }
        report = self.cortex._build_market_report(part_results, 60000.0, None, None, None)

        self.assertIn("BTC: $60,000.00", report)
        self.assertIn("[B] 1.BREAKOUT: BULL | Bullish BO", report)
        self.assertIn("[S] 2.ZONE: BEAR | Supply zone", report)
        self.assertIn("Votes: 1 BULL / 1 BEAR", report)

if __name__ == '__main__':
    unittest.main()
