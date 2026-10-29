from datetime import datetime, timezone
from options_expiry import parse_option_contract, group_current_expiries

def test_future_maturity_is_current_observation():
    assert parse_option_contract('BTC-25SEP26-100000-C', datetime(2026,9,24,tzinfo=timezone.utc))['expiry']=='2026-09-25'

def test_expired_malformed_and_cross_symbol_rejected():
    now=datetime(2026,9,24,tzinfo=timezone.utc)
    assert parse_option_contract('BTC-23SEP26-100000-C',now) is None
    assert parse_option_contract('BTC-bad-100000-C',now) is None
    out=group_current_expiries([{'instrument_name':'ETH-25SEP26-100-C','open_interest':1}], 'BTC', now)
    assert out['expiry_count']==0

def test_ordered_coverage_and_missing_not_invented():
    now=datetime(2026,9,24,tzinfo=timezone.utc)
    out=group_current_expiries([
      {'instrument_name':'BTC-27SEP26-100-C','open_interest':2,'volume':3},
      {'instrument_name':'BTC-25SEP26-100-P','open_interest':4},
      {'instrument_name':'BTC-26SEP26-100-C','volume':1},
    ], 'BTC', now)
    assert [x['expiry'] for x in out['expiries']]==['2026-09-25','2026-09-26','2026-09-27']
    assert out['expiries'][1]['open_interest']==0 and out['expiries'][1]['missing_fields']==1
