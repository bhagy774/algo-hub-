"""Offline behavior checks for Part14's production options math via AST extraction."""
import ast, math, os, unittest, logging, time
from pathlib import Path

class Series:
    def __init__(self, values): self.values=values
    @property
    def iloc(self): return self
    def __getitem__(self, index): return self.values[index]
    def __len__(self): return len(self.values)

SOURCE = (Path(__file__).parent.parent / "jarvis_FIXED.py").read_text(encoding="utf-8")
tree = ast.parse(SOURCE)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Part14OptionsChain")
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"analyze", "analyze_options_deterministically"}]
module = ast.Module(body=methods, type_ignores=[])

class Delta:
    def get_institutional_bias(self, asset):
        return {"bias":"BULLISH", "score":4, "pcr":0.75, "support_wall":95,
                "resistance_wall":110, "max_pain":102, "reasons":["fixture chain"],
                "raw_data":{"source_provider":"Delta", "options_validation":{
                    "underlying":"ETH", "usable":True, "complete":True,
                    "identity_valid":True, "fresh":True},
                    "coverage":{"contract_count":2}}}
class Fake:
    asset="ETH"
    backtest_mode=False
    delta_client=Delta()
    deribit=None
    last_legacy_llm_whale_tag="WHALE_BEARISH"
    last_legacy_llm_insight="stale"
    last_legacy_llm_time=0
    legacy_llm_cooldown=0
    def _generate_legacy_llm_options_prompt(self,*a): raise AssertionError("model path called")

class OptionsMathIntegration(unittest.TestCase):
    def setUp(self):
        old=os.environ.get("JARVIS_PURE_ALGO"); os.environ["JARVIS_PURE_ALGO"]="true"
        self.addCleanup(lambda: os.environ.pop("JARVIS_PURE_ALGO",None) if old is None else os.environ.__setitem__("JARVIS_PURE_ALGO",old))
        calls=[]
        ns={"Dict":dict,"Tuple":tuple,"math":math,"time":time,"logging":logging,
            "_pure_algorithm_mode":lambda:True,"LEGACY_LLM_INTEGRATION_AVAILABLE":True,
            "call_legacy_llm":lambda *a,**k:calls.append(a),"_calls":calls}
        exec(compile(module,"jarvis_FIXED.py","exec"),ns)
        Fake.analyze_options_deterministically = ns["analyze_options_deterministically"]
        self.analyze=ns["analyze"]; self.analyze_whale=ns["analyze_options_deterministically"]; self.calls=calls
    def _run(self, price):
        fake=Fake()
        fake.analyze_options_deterministically=lambda telemetry, current: self.analyze_whale(fake, telemetry, current)
        return self.analyze(fake,{"close":Series([price])},{"symbol":"ETHUSDT"})
    def test_observed_levels_produce_finite_distances_without_model_call(self):
        out=self._run(100.0); t=out["telemetry"]
        self.assertEqual(out["signal"],1)
        self.assertAlmostEqual(t["support_distance_pct"],5.0)
        self.assertAlmostEqual(t["resistance_distance_pct"],10.0)
        self.assertAlmostEqual(t["max_pain_distance_pct"],2.0)
        self.assertEqual(t["math_model"],"observed_chain_descriptive_v1")
        self.assertEqual(t["advisory_status"],"DETERMINISTIC_CHAIN_ONLY")
        self.assertNotIn("legacy_llm_whale_tag", t)
        self.assertEqual(self.calls,[])
    def test_invalid_price_fails_closed(self):
        out=self._run(float('nan'))
        self.assertEqual(out["signal"],0)
        self.assertFalse(out["telemetry"]["available"])
        self.assertEqual(self.calls,[])

    def test_real_no_data_shape_is_not_reported_as_valid_intelligence(self):
        fake = Fake()
        class NoData:
            def get_institutional_bias(self, asset):
                return {"bias":"NEUTRAL", "score":0, "reasons":["No Data"]}
        fake.delta_client = NoData()
        fake.analyze_options_deterministically=lambda telemetry, current: self.analyze_whale(fake, telemetry, current)
        out = self.analyze(fake, {"close":Series([100.0])}, {"selected_symbol":"ETHUSDT"})
        self.assertEqual(out["signal"], 0)
        self.assertFalse(out["telemetry"]["available"])
        self.assertEqual(self.calls,[])

if __name__ == "__main__": unittest.main()
