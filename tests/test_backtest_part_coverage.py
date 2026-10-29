import ast
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BACKTESTER = (HERE / 'jarvis_backtester.py').read_text(encoding='utf-8', errors='ignore')
ENGINE = (HERE / 'jarvis_FIXED.py').read_text(encoding='utf-8', errors='ignore')


def _constant_dict(name):
    tree = ast.parse(BACKTESTER)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f'{name} not declared')


def test_all_twelve_parts_are_declared():
    parts = _constant_dict('CORE_PART_DATA_CONTRACT')
    assert len(parts) == 12
    assert set(parts) == {
        'part1_breakout', 'part2_zone', 'part3_psychology', 'part4_volume',
        'part5_ml', 'part6_trend', 'part7_volatility', 'part8_structure',
        'part9_orderflow', 'part10_candlestats', 'part11_fusion', 'part12_confidence'
    }


def test_coverage_is_written_from_runtime_results():
    assert 'self._record_part_activity(brain)' in BACKTESTER
    assert "'_part_coverage.json'" in BACKTESTER
    assert "'active': stats['decisions_seen'] > 0 and stats['offline'] == 0" in BACKTESTER


def test_backtest_excludes_options_vote_and_rebuilds_local_mtf():
    part_start = ENGINE.index('self.parts = {')
    part_end = ENGINE.index('self.mtf_analyzer', part_start)
    part_block = ENGINE[part_start:part_end]
    assert "if not self.is_backtest_mode:" in part_block
    assert "self.parts['part14_options_chain']" in part_block
    start = ENGINE.index('elif self.is_backtest_mode:', ENGINE.index('MULTI-TIMEFRAME ANALYSIS'))
    end = ENGINE.index('# Synchronize every native live interval', start)
    block = ENGINE[start:end]
    historical_branch = block.split('            else:', 1)[0]
    assert "mtf_data = {'1m': data.copy()}" in historical_branch
    assert '_fetch_mtf_from_api()' not in historical_branch
