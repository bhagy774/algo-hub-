import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

import delta_api_wrapper


class OpenPositionsRequestTests(unittest.TestCase):
    def test_symbol_is_sent_as_signed_product_filter_without_duplicate_params(self):
        wrapper = delta_api_wrapper.DeltaExchangeData.__new__(delta_api_wrapper.DeltaExchangeData)
        wrapper.api_key = "test-key"
        wrapper.api_secret = "test-secret"
        wrapper.session = Mock()

        response = Mock(status_code=200)
        response.json.return_value = {"result": []}
        wrapper.session.request.return_value = response

        with patch.object(wrapper, "_resolve_product",
                          return_value={"id": 123, "symbol": "BTCUSD"}):
            result = wrapper.get_open_positions("BTCUSDT")

        self.assertEqual(result, [])
        wrapper.session.request.assert_called_once()
        method, url = wrapper.session.request.call_args.args[:2]
        request_kwargs = wrapper.session.request.call_args.kwargs
        self.assertEqual(method, "GET")
        self.assertEqual(parse_qs(urlsplit(url).query), {"product_id": ["123"]})
        self.assertNotIn("params", request_kwargs)
        self.assertIn("signature", request_kwargs["headers"])


if __name__ == "__main__":
    unittest.main()


class ProductResolutionTests(unittest.TestCase):
    def _wrapper_with_products(self, symbols):
        wrapper = delta_api_wrapper.DeltaExchangeData.__new__(delta_api_wrapper.DeltaExchangeData)
        wrapper.api_key = "test-key"
        wrapper.api_secret = "test-secret"
        wrapper.session = Mock()
        wrapper._products_cache = {"ts": 9e18, "products": [
            {"id": i + 1, "symbol": s} for i, s in enumerate(symbols)
        ]}
        return wrapper

    def test_usdt_symbol_resolves_to_usd_listing(self):
        wrapper = self._wrapper_with_products(["BTCUSD", "ETHUSD"])
        self.assertEqual(wrapper.get_product_id("BTCUSDT"), "1")
        self.assertEqual(wrapper.get_product_id("ETHUSDT"), "2")

    def test_exact_symbol_still_wins(self):
        wrapper = self._wrapper_with_products(["BTCUSD", "BTCUSDT"])
        self.assertEqual(wrapper.get_product_id("BTCUSDT"), "2")

    def test_unknown_symbol_returns_none(self):
        wrapper = self._wrapper_with_products(["BTCUSD"])
        self.assertIsNone(wrapper.get_product_id("DOGEUSDT"))

    def test_positions_filter_uses_resolved_symbol(self):
        wrapper = self._wrapper_with_products(["BTCUSD"])
        response = Mock(status_code=200)
        response.json.return_value = {"result": [
            {"symbol": "BTCUSD", "size": "1"},
            {"symbol": "ETHUSD", "size": "1"},
        ]}
        wrapper.session.request.return_value = response
        positions = wrapper.get_open_positions("BTCUSDT")
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "BTCUSD")
