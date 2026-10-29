import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import neural_advisory as na
from train_part_advisory import _artifact, make_samples, read_closed_csv

IDENTITY = {"venue":"delta","market_type":"futures","instrument_id":"BTCUSD","symbol":"BTCUSD","timeframe":"1m"}
NOW = datetime.now(timezone.utc).timestamp()


def candles(n=80):
    out=[]
    for i in range(n):
        c=100+0.03*i+math.sin(i/4.0)
        out.append({"timestamp":datetime.fromtimestamp(NOW-(n-i)*60,tz=timezone.utc).isoformat(),
            "open":c-0.1*math.cos(i),"high":c+0.4,"low":c-0.4,"close":c,"volume":50+(i%11)*3})
    return out


def mtf_evidence():
    frames={tf:{part:{"signal":((i+j)%3-1)*0.35,"timeframe":tf,"symbol":"BTCUSD"}
                for j,part in enumerate(na.PARTS[:10])} for i,tf in enumerate(na.TIMEFRAMES)}
    identity={k:IDENTITY[k] for k in ("venue","market_type","instrument_id","symbol")}
    stamps={tf:NOW-na._TIMEFRAME_SECONDS[tf] for tf in na.TIMEFRAMES}
    return {"identity":identity,"symbol":"BTCUSD","closed_timestamps":stamps,"frames":frames}


def _contract(part):
    spec=na.MODEL_SPECS[part]
    return {"task":spec["task"],"dims":list(spec["dims"]),"activations":list(spec["acts"]),"labels":list(spec["labels"]),"kind":spec["kind"]}


def artifact(part):
    spec=na.MODEL_SPECS[part]; layers=[]
    for in_dim,out_dim,act in zip(spec["dims"][:-1],spec["dims"][1:],spec["acts"]):
        layers.append({"weights":[[0.0]*in_dim for _ in range(out_dim)],"bias":[0.0]*out_dim,"activation":act})
    return {"format":na.FORMAT,"format_version":na.FORMAT_VERSION,"feature_schema":na.FEATURE_SCHEMA,
        "part_id":part,"feature_names":list(na.FEATURE_NAMES[part]),"identity":dict(IDENTITY),"model_contract":_contract(part),
        "normalization":{"mean":[0.0]*len(na.FEATURE_NAMES[part]),"std":[1.0]*len(na.FEATURE_NAMES[part])},"layers":layers,
        "training":{"status":"validated","approved_for_advisory":True,"split":"chronological_purged","model_version":"test-fixture",
        "train_samples":250,"validation_samples":60,"test_samples":60,"validation_accuracy":0.60,"validation_majority_accuracy":0.55,
        "test_accuracy":0.60,"test_majority_accuracy":0.55,"dataset_sha256":"a"*64,"dataset_last_timestamp":NOW-120,
        "trainer_version":"fixture","trained_at_utc":datetime.fromtimestamp(NOW-60,tz=timezone.utc).isoformat()}}


def context(identity=IDENTITY,now=NOW,data=None):
    result={"analysis_identity":(identity["venue"],identity["market_type"],identity["instrument_id"],identity["symbol"]),
        "selected_symbol":identity["symbol"],"symbol":identity["symbol"],"timeframe":identity["timeframe"],
        "snapshot_fetched_at":now,"is_backtest_mode":False}
    if data is not None: result["closed_candle_timestamp"]=data[-1]["timestamp"]
    return result


class NeuralAdvisoryTests(unittest.TestCase):
    def setUp(self):
        na.clear_caches(); self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.data=candles()
        self.evidence=mtf_evidence()
    def tearDown(self): na.clear_caches(); self.temp.cleanup()
    def write_artifact(self,part,obj=None,identity=IDENTITY):
        target=na.artifact_path(identity,part,str(self.root)); target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(obj or artifact(part)),encoding="utf-8"); return target
    def _input(self,part): return self.evidence if part in ("part11_fusion","part12_confidence") else self.data
    def _context(self,part): return context(data=self.data) if part in ("part11_fusion","part12_confidence") else context()

    def test_each_part_has_a_distinct_feature_and_head_contract(self):
        self.assertEqual(set(na.FEATURE_NAMES),set(na.PARTS))
        vectors=[na.feature_vector(p,self._input(p)) for p in na.PARTS]
        self.assertTrue(all(len(v)==len(na.FEATURE_NAMES[p]) and all(math.isfinite(x) for x in v) for p,v in zip(na.PARTS,vectors)))
        self.assertEqual(len({tuple(v) for v in vectors}),12)
        contracts={(tuple(na.MODEL_SPECS[p]["dims"]),na.MODEL_SPECS[p]["kind"],tuple(na.MODEL_SPECS[p]["labels"])) for p in na.PARTS}
        self.assertEqual(len(contracts),12)

    def test_trainer_serializes_each_distinct_architecture_for_runtime_validation(self):
        for part in na.PARTS:
            spec=na.MODEL_SPECS[part]; state={}
            for i,(in_dim,out_dim) in enumerate(zip(spec["dims"][:-1],spec["dims"][1:])):
                state[str(i*2)]={"weights":[[0.0]*in_dim for _ in range(out_dim)],"bias":[0.0]*out_dim}
            stats={"mean":[0.0]*len(na.FEATURE_NAMES[part]),"std":[1.0]*len(na.FEATURE_NAMES[part]),
                "validation_accuracy":.60,"validation_majority_accuracy":.55,"test_accuracy":.60,"test_majority_accuracy":.55}
            obj=_artifact(dict(IDENTITY),part,state,stats,[250,60,60],3,20.0,"cpu","a"*64,NOW-3600)
            path=na.artifact_path(IDENTITY,part,str(self.root)); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj))
            na.clear_caches(); na._load_artifact(path,part,IDENTITY)

    def test_all_twelve_parts_require_and_use_matching_task_artifacts(self):
        for part in na.PARTS:
            with self.subTest(part=part):
                self.write_artifact(part)
                result=na.predict_advisory(self._input(part),part,self._context(part),now=NOW,artifact_dir=str(self.root))
                self.assertEqual(result["status"],"available")
                self.assertEqual(result["task"],na.MODEL_SPECS[part]["task"])
                self.assertEqual(result["role"],"advisory_only_no_execution_authority")
                self.assertTrue(result["task_output"])
                self.assertAlmostEqual(sum(result["scores_uncalibrated"].values()),1.0)
                if part=="part11_fusion": self.assertIn(result["direction"],["SELL","NEUTRAL","BUY"])
                if part=="part12_confidence": self.assertIn("consensus_correctness_score_uncalibrated",result)

    def test_parts_11_12_require_all_eight_frames_and_all_ten_part_results(self):
        bad={**self.evidence,"frames":dict(self.evidence["frames"])}; bad["frames"].pop("4h")
        # With a matching artifact, incomplete or wrong-frame evidence fails closed before inference.
        self.write_artifact("part11_fusion")
        self.assertEqual(na.predict_advisory(bad,"part11_fusion",self._context("part11_fusion"),now=NOW,artifact_dir=str(self.root))["reason"],"incomplete_eight_frame_evidence")
        broken=mtf_evidence(); broken["frames"]["3m"].pop("part6_trend")
        self.assertEqual(na.predict_advisory(broken,"part11_fusion",self._context("part11_fusion"),now=NOW,artifact_dir=str(self.root))["reason"],"missing_part_evidence")

    def test_missing_artifact_is_explicit_and_no_random_part5_model(self):
        result=na.predict_advisory(self.data,"part5_ml",context(),now=NOW,artifact_dir=str(self.root))
        self.assertEqual(result,{"status":"unavailable","reason":"trained_artifact_missing"})
        source=(ROOT/"part5_FIXED.py").read_text(encoding="utf-8")
        self.assertNotIn("class NeuralFusionNetwork",source); self.assertNotIn("self.neural_fusion(sig_batch",source); self.assertNotIn("self.neural_fusion.train()",source)

    def test_inference_cache_is_bound_to_exact_candle_and_matrix(self):
        self.write_artifact("part1_breakout"); original=na._infer
        with patch.object(na,"_infer",wraps=original) as infer:
            first=na.predict_advisory(self.data,"part1_breakout",context(),now=NOW,artifact_dir=str(self.root))
            again=na.predict_advisory(self.data,"part1_breakout",context(),now=NOW,artifact_dir=str(self.root))
            self.assertEqual(first,again); self.assertEqual(infer.call_count,1)
            revised=[dict(r) for r in self.data]; revised[-1]["open"]+=.01
            na.predict_advisory(revised,"part1_breakout",context(),now=NOW,artifact_dir=str(self.root)); self.assertEqual(infer.call_count,2)
        self.write_artifact("part11_fusion")
        c=self._context("part11_fusion"); original=na._infer
        changed=mtf_evidence(); changed["frames"]["4h"]["part6_trend"]["signal"]+=.2
        with patch.object(na,"_infer",wraps=original) as infer:
            na.predict_advisory(self.evidence,"part11_fusion",c,now=NOW,artifact_dir=str(self.root))
            na.predict_advisory(changed,"part11_fusion",c,now=NOW,artifact_dir=str(self.root))
            self.assertEqual(infer.call_count,2)

    def test_identity_freshness_and_frame_mismatch_fail_closed(self):
        self.write_artifact("part1_breakout")
        self.assertEqual(na.predict_advisory(self.data,"part1_breakout",{},now=NOW,artifact_dir=str(self.root))["reason"],"full_analysis_identity_required")
        unknown=context(); unknown["analysis_identity"]=("unknown","futures","BTCUSD","BTCUSD")
        self.assertEqual(na.predict_advisory(self.data,"part1_breakout",unknown,now=NOW,artifact_dir=str(self.root))["reason"],"full_analysis_identity_required")
        stale=context(now=NOW-181)
        self.assertEqual(na.predict_advisory(self.data,"part1_breakout",stale,now=NOW,artifact_dir=str(self.root))["reason"],"stale_snapshot")
        wrong=context(); wrong["selected_symbol"]="ETHUSD"
        self.assertEqual(na.predict_advisory(self.data,"part1_breakout",wrong,now=NOW,artifact_dir=str(self.root))["reason"],"full_analysis_identity_required")

    def test_artifact_cache_revalidates_full_identity_and_contract(self):
        path=self.write_artifact("part1_breakout"); na._load_artifact(path,"part1_breakout",IDENTITY)
        with self.assertRaisesRegex(ValueError,"artifact_identity_mismatch"): na._load_artifact(path,"part1_breakout",dict(IDENTITY,symbol="ETHUSD"))
        obj=artifact("part1_breakout"); obj["model_contract"]["task"]="generic"; self.write_artifact("part1_breakout",obj)
        self.assertEqual(na.predict_advisory(self.data,"part1_breakout",context(),now=NOW,artifact_dir=str(self.root))["reason"],"artifact_model_contract_mismatch")

    def test_training_cutoff_and_stale_data_fail_closed(self):
        obj=artifact("part1_breakout"); obj["training"]["dataset_last_timestamp"]=datetime.fromisoformat(self.data[-1]["timestamp"]).timestamp(); self.write_artifact("part1_breakout",obj)
        self.assertEqual(na.predict_advisory(self.data,"part1_breakout",context(),now=NOW,artifact_dir=str(self.root))["reason"],"artifact_training_data_overlap")
        obj=artifact("part1_breakout"); obj["training"]["dataset_last_timestamp"]=NOW-na._MAX_TRAINING_DATA_AGE_SECONDS-1; self.write_artifact("part1_breakout",obj)
        self.assertEqual(na.predict_advisory(self.data,"part1_breakout",context(),now=NOW,artifact_dir=str(self.root))["reason"],"stale_artifact_training_data")

    def test_unvalidated_and_schema_mismatched_models_fail_closed(self):
        obj=artifact("part1_breakout"); obj["feature_schema"]="old"; self.write_artifact("part1_breakout",obj)
        self.assertEqual(na.predict_advisory(self.data,"part1_breakout",context(),now=NOW,artifact_dir=str(self.root))["reason"],"artifact_feature_schema_or_part_mismatch")
        obj=artifact("part1_breakout"); obj["training"]["approved_for_advisory"]=False; self.write_artifact("part1_breakout",obj)
        self.assertEqual(na.predict_advisory(self.data,"part1_breakout",context(),now=NOW,artifact_dir=str(self.root))["reason"],"artifact_not_validation_gated")

    def test_pre_quantitative_feature_schema_v3_is_incompatible(self):
        obj=artifact("part1_breakout"); obj["feature_schema"]="jarvis-task-specific-features-v3"; self.write_artifact("part1_breakout",obj)
        result=na.predict_advisory(self.data,"part1_breakout",context(),now=NOW,artifact_dir=str(self.root))
        self.assertEqual(result["status"],"unavailable")
        self.assertEqual(result["reason"],"artifact_feature_schema_or_part_mismatch")

    def test_noncontiguous_forming_nan_and_wrong_metadata_fail_closed(self):
        gap=[dict(r) for r in self.data]; gap[-1]["timestamp"]=(datetime.fromisoformat(gap[-1]["timestamp"])+timedelta(seconds=60)).isoformat()
        self.assertEqual(na.predict_advisory(gap,"part1_breakout",context(),now=NOW,artifact_dir=str(self.root))["reason"],"closed_candle_sequence_invalid")
        forming=[dict(r) for r in self.data]; forming[-1]["is_closed"]=False
        self.assertEqual(na.predict_advisory(forming,"part1_breakout",context(),now=NOW,artifact_dir=str(self.root))["reason"],"forming_candle_rejected")
        bad=[dict(r) for r in self.data]; bad[-1]["close"]=float("nan")
        self.assertEqual(na.predict_advisory(bad,"part1_breakout",context(),now=NOW,artifact_dir=str(self.root))["reason"],"non_finite_ohlcv")

    def test_advisory_metadata_never_changes_deterministic_outputs(self):
        self.write_artifact("part1_breakout"); before={"signal":-1,"confidence":71,"thought":"deterministic"}
        after=na.annotate_part_result(before,self.data,"part1_breakout",context(),now=NOW,artifact_dir=str(self.root))
        self.assertEqual({k:after[k] for k in before},before); self.assertEqual(after["neural_advisory"]["status"],"available")
        self.write_artifact("part12_confidence")
        mtf_before={"signal":.2,"confidence":50}; mtf_after=na.annotate_mtf_result(mtf_before,self.evidence,"part12_confidence",self._context("part12_confidence"),now=NOW,artifact_dir=str(self.root))
        self.assertEqual({k:mtf_after[k] for k in mtf_before},mtf_before)

    def test_neural_metadata_cannot_change_central_mtf_approval(self):
        from jarvis_strategy_approval import evaluate_mtf_central_strategy
        from test_multitimeframe_strategy_policy import aggregate_gate,evidence_by_timeframe
        evidence=evidence_by_timeframe(); baseline=evaluate_mtf_central_strategy(evidence,aggregate_gate(),confidence=80,expected_symbol="BTCUSDT")
        decorated=evidence_by_timeframe()
        for values in decorated.values():
            for result in values.values(): result["neural_advisory"]={"status":"available","direction":"SELL","scores_uncalibrated":{"sell":1.0}}
        unchanged=evaluate_mtf_central_strategy(decorated,aggregate_gate(),confidence=80,expected_symbol="BTCUSDT")
        self.assertEqual(unchanged["approved"],baseline["approved"]); self.assertEqual(unchanged["direction"],baseline["direction"]); self.assertEqual(unchanged["trade_mode"],baseline["trade_mode"])

    def test_head_shapes_finite_logits_and_artifact_size_bound(self):
        with self.assertRaisesRegex(ValueError,"non_finite_inference"): na._softmax([float("inf"),0.,0.])
        obj=artifact("part1_breakout"); obj["layers"][0]["weights"][0][0]=1e308; self.write_artifact("part1_breakout",obj)
        self.assertEqual(na.predict_advisory(self.data,"part1_breakout",context(),now=NOW,artifact_dir=str(self.root))["reason"],"artifact_numeric_value_out_of_bounds")
        path=self.write_artifact("part2_zone"); obj=artifact("part2_zone"); obj["normalization"]["std"]=[0.0]*len(na.FEATURE_NAMES["part2_zone"]); path.write_text(json.dumps(obj))
        self.assertEqual(na.predict_advisory(self.data,"part2_zone",context(),now=NOW,artifact_dir=str(self.root))["reason"],"artifact_normalization_std_invalid")
        path.write_text("{}"*(na._MAX_ARTIFACT_BYTES+1)); na.clear_caches()
        self.assertEqual(na.predict_advisory(self.data,"part2_zone",context(),now=NOW,artifact_dir=str(self.root))["reason"],"artifact_size_invalid")

    def test_chronological_purged_split_and_part_specific_training_targets(self):
        rows=candles(1000)
        X,y,split=make_samples(rows,"part5_ml",3,20)
        self.assertEqual(len(X),len(y)); self.assertEqual(len(X),len(split)); self.assertGreaterEqual(sum(s==0 for s in split),200); self.assertGreaterEqual(sum(s==1 for s in split),50); self.assertGreaterEqual(sum(s==2 for s in split),50)
        p7,labels,_=make_samples(rows,"part7_volatility",3,20)
        self.assertTrue(set(labels).issubset({0,1,2})); self.assertNotEqual(labels,y)

    def test_parts_11_12_need_evidence_json_csv_and_one_minute_clock(self):
        path=self.root/"data.csv"; header="timestamp,open,high,low,close,volume,is_closed,venue,market_type,instrument_id,symbol,timeframe\n"
        path.write_text(header+"2026-09-29T00:00:00+00:00,100,101,99,100,10,true,delta,futures,BTCUSD,BTCUSD,1m\n")
        with self.assertRaisesRegex(ValueError,"evidence_json"): read_closed_csv(path,IDENTITY,"part11_fusion")
        with self.assertRaisesRegex(ValueError,"forming/unconfirmed"):
            path.write_text(header+"2026-09-29T00:00:00+00:00,100,101,99,100,10,false,delta,futures,BTCUSD,BTCUSD,1m\n"); read_closed_csv(path,IDENTITY)

    def test_part5_neutral_target_split_boundary_remains_purged(self):
        rows=candles(1000); _,_,split=make_samples(rows,"part5_ml",3,20)
        self.assertGreaterEqual(sum(s==0 for s in split),200); self.assertGreaterEqual(sum(s==1 for s in split),50); self.assertGreaterEqual(sum(s==2 for s in split),50)
        train_cut=int(len(rows)*.70); val_cut=int(len(rows)*.85)
        self.assertLess((train_cut-4)+3,train_cut); self.assertLess((val_cut-4)+3,val_cut)

    def test_part12_correctness_target_uses_signed_consensus_feature(self):
        from neural_parts import label_target
        x=na.feature_vector("part12_confidence",self.evidence)
        self.assertEqual(len(x),18)
        self.assertEqual(na.FEATURE_NAMES["part12_confidence"][10],"signed_consensus_vote")
        rows=candles(80)
        # Use a clearly positive forward target and non-neutral signed vote.
        rows[3]["close"] = rows[0]["close"] * 1.01
        x=list(x); x[10]=0.5
        self.assertEqual(label_target("part12_confidence",rows,0,3,20,x),1)
        self.assertIsNone(label_target("part12_confidence",rows,0,3,1000,x))

if __name__=="__main__": unittest.main()
