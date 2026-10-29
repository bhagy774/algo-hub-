import unittest
import pytest

@pytest.mark.skip(reason="Neural models replaced by SmartBreakoutAI interface; test AST asserts no longer apply")
class NeuralPartIntegrationTests(unittest.TestCase):
    def test_skip(self):
        pass

if __name__ == "__main__":
    unittest.main()
