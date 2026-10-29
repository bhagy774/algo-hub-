from jarvis_position_manager import PositionRequest
import unittest
from unittest.mock import MagicMock
from jarvis_position_manager import JarvisPositionManager, PositionRecord, get_position_manager

class TestPositionRecord(unittest.TestCase):
    def test_initialization(self):
        record = PositionRecord(
            position_id="test_pos_1",
            direction="CALL",
            entry_price=50000.0,
            contracts=2,
            confidence=85,
            coin="BTC",
            trade_type="SCALP",
            contract_value_usdt=1.0,
            notional_usdt=2.0
        )
        self.assertEqual(record.id, "test_pos_1")
        self.assertEqual(record.direction, "CALL")
        self.assertEqual(record.entry_price, 50000.0)
        self.assertEqual(record.contracts, 2)
        self.assertEqual(record.confidence, 85)
        self.assertEqual(record.coin, "BTC")
        self.assertEqual(record.trade_type, "SCALP")
        self.assertEqual(record.contract_value_usdt, 1.0)
        self.assertEqual(record.notional_usdt, 2.0)
        self.assertTrue(record.is_call)

    def test_current_pnl_pct_call(self):
        record = PositionRecord(
            position_id="test_pos_2",
            direction="CALL",
            entry_price=100.0,
            contracts=1,
            confidence=90,
            coin="ETH",
            contract_value_usdt=1.0
        )
        self.assertAlmostEqual(record.current_pnl_pct(110.0), 0.1)
        self.assertAlmostEqual(record.current_pnl_pct(90.0), -0.1)

    def test_current_pnl_pct_put(self):
        record = PositionRecord(
            position_id="test_pos_3",
            direction="PUT",
            entry_price=100.0,
            contracts=1,
            confidence=90,
            coin="ETH",
            contract_value_usdt=1.0
        )
        self.assertAlmostEqual(record.current_pnl_pct(90.0), 0.1)
        self.assertAlmostEqual(record.current_pnl_pct(110.0), -0.1)

    def test_invalid_contract_value(self):
        with self.assertRaises(ValueError):
            PositionRecord(
                position_id="test_pos_4",
                direction="CALL",
                entry_price=100.0,
                contracts=1,
                confidence=90,
                contract_value_usdt=0.0
            )

class TestJarvisPositionManager(unittest.TestCase):
    def setUp(self):
        self.delta_mock = MagicMock()
        self.manager = JarvisPositionManager(delta_client=self.delta_mock)
        # reset singleton for testing get_position_manager cleanly if needed, though get_position_manager handles it.
        import jarvis_position_manager
        jarvis_position_manager._pm_instance = None

    def tearDown(self):
        self.manager.stop()
        import jarvis_position_manager
        jarvis_position_manager._pm_instance = None

    def test_initialization(self):
        self.assertIsNotNone(self.manager.delta)
        self.assertFalse(self.manager._running)
        self.assertEqual(self.manager.get_open_count(), 0)

    def test_singleton(self):
        pm1 = get_position_manager(delta_client=self.delta_mock)
        pm2 = get_position_manager(delta_client=self.delta_mock)
        self.assertIs(pm1, pm2)

    def test_register_position_success(self):
        pos = self.manager.register_position(PositionRequest(position_id="reg_pos_1", direction="CALL", entry_price=50000.0, contracts=1, confidence=80, coin="BTC", contract_value_usdt=1.0))
        self.assertIsInstance(pos, PositionRecord)
        self.assertEqual(self.manager.get_open_count(), 1)
        self.assertTrue(self.manager.has_open_position())

    def test_register_position_invalid_id(self):
        with self.assertRaises(ValueError):
            self.manager.register_position(PositionRequest(position_id="", direction="CALL", entry_price=50000.0, contracts=1, confidence=80, coin="BTC", contract_value_usdt=1.0))

    def test_register_position_invalid_direction(self):
        with self.assertRaises(ValueError):
            self.manager.register_position(PositionRequest(position_id="reg_pos_inv_dir", direction="INVALID", entry_price=50000.0, contracts=1, confidence=80, coin="BTC", contract_value_usdt=1.0))

    def test_register_position_invalid_price(self):
        with self.assertRaises(ValueError):
            self.manager.register_position(PositionRequest(position_id="reg_pos_inv_price", direction="CALL", entry_price=-100.0, contracts=1, confidence=80, coin="BTC", contract_value_usdt=1.0))

    def test_status_line(self):
        status = self.manager.status_line()
        self.assertIn("POSITION MGR:", status)
        self.assertIn("Open=", status)

if __name__ == '__main__':
    unittest.main()
