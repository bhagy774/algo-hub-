"""Pure mode prevents optional hedge-advisor/monitor model calls."""
import unittest, os
from options_hedged_scalp import OptionsHedgedScalpEngine

class Advisor:
    def __init__(self): self.calls=[]
    def evaluate_hedge_setup(self, **kwargs): self.calls.append("evaluate"); return {"hedge":"YES","strike":100,"hedge_ratio":1}
    def monitor_active_hedge(self, **kwargs): self.calls.append("monitor"); return {}

class OptionsAIGuardTests(unittest.TestCase):
    def test_pure_mode_skips_advisor_and_keeps_paper_futures_path(self):
        old=os.environ.get("JARVIS_PURE_ALGO"); os.environ["JARVIS_PURE_ALGO"]="true"
        self.addCleanup(lambda: os.environ.pop("JARVIS_PURE_ALGO",None) if old is None else os.environ.__setitem__("JARVIS_PURE_ALGO",old))
        advisor=Advisor(); engine=OptionsHedgedScalpEngine(None, advisor, None)
        engine.start_monitor=lambda: None
        result=engine.execute_hedged_scalp({"direction":"BUY","confidence":75},100,2,{"puts":[{"symbol":"P","strike":100,"price":2}]})
        self.assertEqual(result["status"],"executed")
        self.assertFalse(result["hedge_applied"])
        self.assertEqual(advisor.calls,[])
        self.assertEqual(len(engine.active_positions),1)

if __name__ == "__main__": unittest.main()
