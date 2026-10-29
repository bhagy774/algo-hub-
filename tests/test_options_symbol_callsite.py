import ast
from pathlib import Path

SOURCE_PATH = Path(__file__).parent.parent / "jarvis_FIXED.py"
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")


def test_production_part14_callsite_passes_routed_symbol_with_isolated_input_and_context():
    tree = ast.parse(SOURCE)
    analysis_fn = next(
        node for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "JarvisElite"
        for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "analyze_trade_setup"
    )
    callsite = ast.get_source_segment(SOURCE, analysis_fn)
    # All adapters receive the selected route and native timeframe; Part14's
    # selected-asset path therefore cannot fall back to its BTC default.
    assert "'selected_symbol': getattr(self, 'active_symbol', None)" in callsite
    assert "'symbol': getattr(self, 'active_symbol', None)" in callsite
    assert "'timeframe': tf_name" in callsite
    # Preserve #95's per-adapter isolation and snapshot identity metadata.
    assert "'analysis_identity': getattr(candle_snapshot, 'identity', None)" in callsite
    assert "part_context = dict(self.market_context)" in callsite
    assert "frame.copy(deep=True)" in callsite
    assert "part_data = tf_data.copy(deep=True)" in callsite
    assert "res = part.analyze(part_data, context=part_context)" in callsite

    part14 = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Part14OptionsChain")
    analyze = next(node for node in part14.body if isinstance(node, ast.FunctionDef) and node.name == "analyze")
    code = ast.get_source_segment(SOURCE, analyze)
    assert "context.get('symbol')" in code
    assert "candidate.get_institutional_bias(query)" in code


def test_oracle_altcoin_block_and_live_safety_guards_are_not_removed():
    root = Path(__file__).parent.parent
    gate = (root / "oracle_trade_gate.py").read_text(encoding="utf-8")
    assert "requested_base != \"BTC\"" in gate
    assert "Oracle BTC macro forecast unavailable for selected asset" in gate
    live = (root / "jarvis_live_trader.py").read_text(encoding="utf-8")
    assert "require_options=bool(self.is_enabled)" in live
