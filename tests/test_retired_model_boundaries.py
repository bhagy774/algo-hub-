"""Offline tests: real lightweight modules plus AST-isolated heavy model class methods."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def isolated_method(file, cls, method):
    """Compile a production method without importing torch or the live trading system."""
    tree = ast.parse((ROOT / file).read_text(encoding='utf-8'))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    function = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == method)
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    ns = {}
    exec(compile(module, file, 'exec'), ns)
    return ns[method]


class RetiredModelBoundaries(unittest.TestCase):
    def test_part2_predict_and_boost_are_no_vote(self):
        forbidden = Mock()
        forbidden.predict.side_effect = AssertionError('model inference')
        owner = Mock(neural_network_manager=forbidden)
        self.assertEqual(isolated_method('part2_FIXED.py', 'NeuralNetworkManager', 'predict')(owner, [1], [1]), {})
        signals = [('CALL', 5., 'math')]
        self.assertIs(isolated_method('part2_FIXED.py', 'AdvancedAnalysisSystem', '_enhance_with_ai')(owner, signals, Mock()), signals)
        forbidden.predict.assert_not_called()

    def test_core_training_and_prediction_are_unavailable(self):
        owner = Mock(models={'random_forest': Mock()})
        train = isolated_method('jarvis_FIXED.py', 'AutoTrainingEngine', 'train_models')
        predict = isolated_method('jarvis_FIXED.py', 'AutoTrainingEngine', 'predict_signal')
        self.assertEqual(train(owner, [[1, 2]])['status'], 'unavailable')
        self.assertEqual(predict(owner, [1, 2]), 0)
        owner.models['random_forest'].predict.assert_not_called()

    def test_doctor_key_does_not_trigger_model_request_or_auto_fix(self):
        from jarvis_doctor import DoctorMonitor, DoctorIssue
        doctor = DoctorMonitor.__new__(DoctorMonitor)
        issue = DoctorIssue('part1', 'CRITICAL', 'test', 'error')
        with patch('jarvis_doctor.DOCTOR_API_KEY', 'configured-secret'), \
             patch('requests.post', side_effect=AssertionError('network model call')) as post:
            self.assertIsNone(doctor._diagnose_with_gemini(issue))
            post.assert_not_called()





if __name__ == '__main__':
    unittest.main()
