import unittest
import io
import sys
from unittest.mock import patch

from professional_display import (
    ProfessionalSignalDisplay,
    _strip,
    _pad,
    _bar,
    _center_visual
)

class TestProfessionalDisplayHelpers(unittest.TestCase):
    def test_strip(self):
        colored_text = '\033[91mHello\033[0m'
        self.assertEqual(_strip(colored_text), 'Hello')

    def test_pad(self):
        colored_text = '\033[91mHello\033[0m'
        padded = _pad(colored_text, 10)
        # Visual length should be 10: "Hello" (5) + 5 spaces.
        # Total string length = length of ansi codes + 10.
        # \033[91m is 5 chars, \033[0m is 4 chars -> 9 chars
        # Total = 19
        self.assertEqual(len(padded), len(colored_text) + 5)
        self.assertEqual(_strip(padded), 'Hello     ')

    def test_bar(self):
        # 50% of width 10 -> 5 filled, 5 empty
        bar_str = _bar(50, width=10)
        self.assertIn('█' * 5, bar_str)
        self.assertIn('░' * 5, bar_str)

    def test_center_visual(self):
        colored_text = '\033[91mHi\033[0m'
        centered = _center_visual(colored_text, 6)
        self.assertEqual(_strip(centered), '  Hi  ')

class TestProfessionalSignalDisplay(unittest.TestCase):
    def setUp(self):
        self.display = ProfessionalSignalDisplay()

    @patch('os.system')
    def test_display_full_signal(self, mock_system):
        signal_data = {
            'direction': 'CALL',
            'confidence': 85,
            'ai_reason': 'Bullish divergence detected',
            'entry_price': 100.50,
            'market_context': {
                'trend': 'UPTREND',
                'volatility': 'NORMAL',
                'session': 'NY'
            },
            'trade_signal': {
                'take_profit_1': 110.0,
                'stop_loss': 95.0,
                'recommended_expiry': '3M'
            }
        }

        part_results = {
            'part1_breakout': {'signal': 1, 'thought': 'Breakout up'}
        }

        captured_output = io.StringIO()
        with patch('sys.stdout', new=captured_output):
            self.display.display_full_signal(
                signal_data=signal_data,
                current_price=101.0,
                part_results=part_results,
                symbol='BTCUSDT',
                model_name='test-model'
            )

        output = captured_output.getvalue()

        self.assertIn('CALL', output)
        self.assertIn('101.00', output)
        self.assertIn('BTCUSDT', output)
        self.assertIn('test-model', output)
        self.assertIn('UPTREND', output)
        self.assertIn('Bullish divergence detected', output)

    @patch('os.system')
    def test_display_full_signal_no_trade(self, mock_system):
        signal_data = {
            'direction': 'NO_TRADE',
            'no_trade_reason': 'Waiting for better setup'
        }

        captured_output = io.StringIO()
        with patch('sys.stdout', new=captured_output):
            self.display.display_full_signal(
                signal_data=signal_data,
                symbol='ETHUSDT'
            )

        output = captured_output.getvalue()

        self.assertIn('WAIT', output)
        self.assertIn('Waiting for better setup', output)

    def test_display_compact(self):
        final_decision = {
            'direction': 'BUY',
            'confidence': 90,
            'entry': 50000.0
        }

        captured_output = io.StringIO()
        with patch('sys.stdout', new=captured_output):
            self.display.display_compact(final_decision)

        output = captured_output.getvalue()
        self.assertIn('BUY', output)
        self.assertIn('90%', output)
        self.assertIn('50,000', output)

if __name__ == '__main__':
    unittest.main()
