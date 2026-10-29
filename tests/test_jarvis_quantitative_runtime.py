"""Offline tests of modified Jarvis adapters and bounded quantitative risk paths."""
from __future__ import annotations
import ast
import copy
import logging
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import quantitative_math as qm
import pandas as pd


def bars(n=100):
    out=[]; price=100.0
    for i in range(n):
        ret=0.0004*math.sin(i*.41)+0.00005
        op=price; price=op*math.exp(ret)
        out.append({"open":op,"high":max(op,price)*1.0007,"low":min(op,price)*.9993,
                    "close":price,"volume":100.0+(i%9)*7})
    return pd.DataFrame(out)


def extract(path, class_name, method_name, globals_):
    source=Path(path).read_text(encoding="utf-8")
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method_name)
    method=copy.deepcopy(method); method.decorator_list=[]
    wrapper=ast.ClassDef(name="Harness",bases=[],keywords=[],body=[method],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]))
    exec(compile(module,str(path),"exec"),globals_)
    return globals_["Harness"]


class JarvisQuantitativeRuntimeTests(unittest.TestCase):
    def test_native_part1_adapter_preserves_analyzer_result_and_forwards_frame(self):
        class Engine:
            def __init__(self): self.args=None
            def analyze(self,*args,**kwargs):
                self.args=(args,kwargs)
                return {"signal":-1,"confidence":62,"breakout":{"breakout_detected":True,"direction":-1,"strength":0.9}, "levels":{"unchanged":True}}
        engine=Engine(); ns={"logger":logging.getLogger("quant-test"),"_df_to_market_data":lambda d:{"price_action":d.to_dict("records")}}
        Harness=extract(ROOT/"jarvis_FIXED.py","Part1Breakout","analyze",ns)
        obj=object.__new__(Harness); obj._engine=engine
        data=bars(); context={"timeframe":"1m"}
        result=obj.analyze(data,context)
        self.assertEqual(result["signal"], -1)
        self.assertEqual(result["telemetry"]["levels"], {"unchanged":True})
        self.assertEqual(engine.args[0][0]["price_action"], data.to_dict("records"))
        self.assertEqual(engine.args[1], {})

    def test_quantitative_fallbacks_for_jarvis_parts_1_to_10(self):
        data=bars()
        cases=[(1,"Part1Breakout"),(3,"Part3Psychology"),(4,"Part4Volume"),
               (5,"Part5ML"),(6,"Part6Trend"),(8,"Part8Structure"),
               (9,"Part9Orderflow"),(10,"Part10Candlestats")]
        for part,cls_name in cases:
            with self.subTest(part=part):
                ns={"logger":logging.getLogger("quant-test")}
                Harness=extract(ROOT/"jarvis_FIXED.py",cls_name,"analyze",ns)
                obj=object.__new__(Harness)
                obj._engine=None
                result=obj.analyze(data,context={"timeframe":"1m"})
                self.assertIn(result.get("signal"),(-1,0,1))
                self.assertTrue(math.isfinite(float(result.get("confidence",5.0))))
                self.assertIsInstance(result.get("thought"), str)
        # The native Part 2 adapter uses the same structural math on its own frame.
        ns={"logger":logging.getLogger("quant-test"),"quantitative_math":qm}
        Harness=extract(ROOT/"jarvis_FIXED.py","Part2Zone","analyze",ns)
        obj=object.__new__(Harness); obj._engine=None
        obj._native_zone=lambda frame,tf:qm.part_signal("2",frame)
        obj._shared_mtf_context=lambda context,version:{"signal":0,"thought":""}
        result=obj.analyze(data,{"timeframe":"1m","snapshot_version":"snap"})
        self.assertIsInstance(result.get("thought"), str)
        self.assertIn(result.get("signal"),(-1,0,1))

    def test_smart_entry_ignores_legacy_atr_and_respects_existing_pullback_bounds(self):
        ns={"normalize_confidence":lambda v:float(str(v).split("/")[0]),
            "logger":logging.getLogger("quant-test")}
        Harness=extract(ROOT/"jarvis_FIXED.py","LiveTradingEngine","_calculate_smart_entry",ns)
        obj=object.__new__(Harness); data=bars()
        base={"market_context":{"volatility":"NORMAL"},"trade_signal":{"confidence_score":"60/100","atr":0.01}}
        corrupt_atr={"market_context":{"volatility":"NORMAL"},"trade_signal":{"confidence_score":"60/100","atr":1000000}}
        a=obj._calculate_smart_entry("CALL",100.0,base)
        b=obj._calculate_smart_entry("CALL",100.0,corrupt_atr)
        # Active policy uses bounded ATR pullback; both inputs must obey its risk bounds.
        self.assertGreaterEqual((100.0-b[0])/100.0,0.0005)
        self.assertLessEqual((100.0-b[0])/100.0,0.004 + 1e-12)
        pullback=(100.0-a[0])/100.0
        self.assertGreaterEqual(pullback,0.0005 - 1e-12)
        self.assertLessEqual(pullback,0.004)
        self.assertEqual(obj._calculate_smart_entry("NO_TRADE",100.0,base)[0],100.0)

    def test_scalping_targets_keep_price_unit_stop_and_take_profit_caps(self):
        ns={"logger":logging.getLogger("quant-test"),"math":math}
        Harness=extract(ROOT/"jarvis_FIXED.py","ScalpingEngine","calculate_targets",ns)
        obj=object.__new__(Harness); data=bars()
        for direction in ("CALL","PUT"):
            with self.subTest(direction=direction):
                result=obj.calculate_targets(data,direction,100.0,
                    options_data={"support":80.0,"resistance":120.0,"max_pain":150.0 if direction=="CALL" else 50.0})
                self.assertIsNotNone(result)
                sl=result["stop_loss"]; tp1=result["take_profit_1"]; tp2=result["take_profit_2"]
                if direction=="CALL":
                    self.assertGreaterEqual(sl,99.60); self.assertLess(sl,100.0)
                    self.assertGreaterEqual(tp1,100.0); self.assertLessEqual(tp1,100.80)
                    self.assertGreaterEqual(tp2,tp1); self.assertLessEqual(tp2,101.50)
                else:
                    self.assertLessEqual(sl,100.40); self.assertGreater(sl,100.0)
                    self.assertLessEqual(tp1,100.0); self.assertGreaterEqual(tp1,99.20)
                    self.assertLessEqual(tp2,tp1); self.assertGreaterEqual(tp2,98.50)
        self.assertIsNone(obj.calculate_targets(data[:10],"CALL",100.0))

if __name__=="__main__": unittest.main()
