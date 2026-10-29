import logging
import unittest
from unittest.mock import Mock, patch
import delta_api_wrapper

class DiagnosticTests(unittest.TestCase):
    def response(self, body, status=401, text=''):
        r = Mock(status_code=status, text=text)
        r.json.return_value = body
        return r

    def test_safe_code_preserved_without_body_leak(self):
        secret = 'dummy-secret-123'
        r = self.response({'error': 'INVALID_API_KEY'}, text=secret)
        result = delta_api_wrapper.DeltaExchangeData._diagnostic_error(r)
        self.assertEqual(result, 'HTTP 401: INVALID_API_KEY')
        self.assertNotIn(secret, result)

    def test_unstructured_and_malformed_codes_rejected(self):
        for body in ({'error': 'bad credential dummy-secret'}, {'error': 'invalid-code'}, {'message': 'dummy-secret'}, 'dummy-secret'):
            r = self.response(body, status=400, text='dummy-secret')
            self.assertEqual(delta_api_wrapper.DeltaExchangeData._diagnostic_error(r), 'HTTP 400')

    def test_request_log_omits_query_and_body(self):
        obj = delta_api_wrapper.DeltaExchangeData.__new__(delta_api_wrapper.DeltaExchangeData)
        obj.session = Mock()
        obj.api_key = 'key'
        obj.api_secret = 'secret'
        obj.session.request.return_value = self.response({'error': 'RATE_LIMIT'}, 429, 'dummy-secret')
        with self.assertLogs(delta_api_wrapper.logger, level='ERROR') as logs:
            result = obj._request('GET', '/v2/tickers?api_secret=dummy-secret', authorized=False)
        self.assertEqual(result['error'], 'HTTP 429: RATE_LIMIT')
        self.assertTrue(any('/v2/tickers' in line and '?' not in line for line in logs.output))
        self.assertFalse(any('dummy-secret' in line for line in logs.output))

if __name__ == '__main__':
    unittest.main()
