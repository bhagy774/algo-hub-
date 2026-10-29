#!/usr/bin/env python3
"""Offline trainer for twelve independent, task-specific Jarvis advisory heads.

No bot, exchange or network calls are made. Each Part has a distinct feature
contract, model topology/head and label definition. Artifacts are accepted only
after chronological validation/test gates; labels are research proxies, not PnL,
calibration, profitability or authorization. PyTorch is used only offline.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neural_advisory import (FEATURE_NAMES, FEATURE_SCHEMA, FORMAT, FORMAT_VERSION,
    MODEL_SPECS, TIMEFRAMES, _TIMEFRAME_SECONDS, artifact_path, feature_vector,
    validate_mtf_evidence)
from neural_parts import label_target

MAX_ROWS = 100_000
MIN_TRAIN = 200
MIN_HOLDOUT = 50


def parse_timestamp(value: str) -> float:
    text = str(value or "").strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _identity_arg(args: argparse.Namespace) -> Dict[str, str]:
    vals = {"venue": args.venue.strip().lower(), "market_type": args.market_type.strip().lower(),
            "instrument_id": args.instrument_id.strip(), "symbol": args.symbol.strip().upper(),
            "timeframe": args.timeframe.strip()}
    if not all(vals.values()) or vals["venue"] in ("unknown", "none", "null") or vals["market_type"] in ("unknown", "none", "null"):
        raise ValueError("Training unavailable: full known venue/market/instrument/symbol/timeframe identity is required.")
    return vals


def read_closed_csv(path: Path, identity: Dict[str, str], part_id: Optional[str] = None) -> List[Dict[str, Any]]:
    expected = dict(identity)
    if part_id in ("part11_fusion", "part12_confidence") and identity["timeframe"] != "1m":
        raise ValueError("Parts 11/12 use one identity-bound 1m decision clock and the complete eight-frame evidence matrix.")
    records: List[Dict[str, Any]] = []
    last_ts = None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"timestamp", "open", "high", "low", "close", "volume", "is_closed",
                        "venue", "market_type", "instrument_id", "symbol", "timeframe"}
            if part_id in ("part11_fusion", "part12_confidence"):
                required.add("evidence_json")
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError("CSV needs chronological closed OHLCV + exact identity; Parts 11/12 also require evidence_json with all 8 frames × Parts 1–10.")
            for line, row in enumerate(reader, start=2):
                if len(records) >= MAX_ROWS:
                    raise ValueError(f"CSV exceeds {MAX_ROWS} rows; split into one bounded identity dataset.")
                if str(row["is_closed"]).strip().lower() not in ("1", "true", "yes"):
                    raise ValueError(f"line {line}: forming/unconfirmed candle; only closed candles are trainable.")
                actual = {"venue": str(row["venue"]).strip().lower(),
                          "market_type": str(row["market_type"]).strip().lower(),
                          "instrument_id": str(row["instrument_id"]).strip(),
                          "symbol": str(row["symbol"]).strip().upper(),
                          "timeframe": str(row["timeframe"]).strip()}
                if actual != expected:
                    raise ValueError(f"line {line}: mixed/wrong instrument identity; dataset must match all CLI identity values.")
                ts = parse_timestamp(row["timestamp"])
                if last_ts is not None:
                    if ts <= last_ts:
                        raise ValueError(f"line {line}: timestamps must be strictly increasing and unique.")
                    if abs((ts - last_ts) - _TIMEFRAME_SECONDS[identity["timeframe"]]) > 2.0:
                        raise ValueError(f"line {line}: gap/irregular interval; contiguous native timeframe candles are required.")
                last_ts = ts
                vals = {k: float(row[k]) for k in ("open", "high", "low", "close", "volume")}
                if not all(math.isfinite(v) for v in vals.values()):
                    raise ValueError(f"line {line}: non-finite OHLCV.")
                if min(vals[k] for k in ("open", "high", "low", "close")) <= 0 or vals["volume"] < 0 or vals["high"] < max(vals["open"], vals["close"], vals["low"]) or vals["low"] > min(vals["open"], vals["close"], vals["high"]):
                    raise ValueError(f"line {line}: invalid OHLCV values.")
                vals["timestamp"] = row["timestamp"]
                if "evidence_json" in required:
                    try:
                        evidence = json.loads(row["evidence_json"])
                        # Training-only timestamp validity, identity binding and
                        # feature checks use the exact runtime evidence contract.
                        validate_mtf_evidence(evidence, identity, ts, is_backtest=True)
                        feature_vector(part_id, evidence)
                    except Exception as exc:
                        raise ValueError(f"line {line}: invalid/incomplete eight-frame Part evidence: {exc}") from exc
                    vals["evidence"] = evidence
                records.append(vals)
    except FileNotFoundError as e:
        raise ValueError(f"Training unavailable: dataset not found: {path}") from e
    return records


def _returns(rows: Sequence[Dict[str, Any]], i: int, horizon: int) -> float:
    return rows[i + horizon]["close"] / rows[i]["close"] - 1.0


def make_samples(rows: Sequence[Dict[str, Any]], part_id: str, horizon: int,
                 neutral_bps: float) -> Tuple[List[List[float]], List[int], List[int]]:
    if len(rows) < 400:
        raise ValueError("Training unavailable: at least 400 strictly identified closed candles are required before chronological holdouts.")
    n = len(rows); cut_train, cut_val = int(n*.70), int(n*.85)
    indices = ([i for i in range(63, max(63, cut_train-horizon))]
               + [i for i in range(cut_train+horizon, max(cut_train+horizon, cut_val-horizon))]
               + [i for i in range(cut_val+horizon, n-horizon)])
    X: List[List[float]]=[]; y: List[int]=[]; split: List[int]=[]
    for i in indices:
        evidence = rows[i].get("evidence")
        if part_id in ("part11_fusion", "part12_confidence"):
            if evidence is None: raise ValueError("Training unavailable: Parts 11/12 require point-in-time evidence_json for every sample.")
            vector = feature_vector(part_id, evidence)
        else:
            vector = feature_vector(part_id, rows[max(0, i-63):i+1])
        label = label_target(part_id, rows, i, horizon, neutral_bps, vector)
        if label is None: continue
        X.append(list(vector)); y.append(label); split.append(0 if i<cut_train else 1 if i<cut_val else 2)
    counts=[sum(s==j for s in split) for j in range(3)]
    if counts[0] < MIN_TRAIN or counts[1] < MIN_HOLDOUT or counts[2] < MIN_HOLDOUT:
        raise ValueError(f"Training unavailable: chronological split too small after {horizon}-bar purge/target filtering (train/validation/test={counts}). Supply more historical data.")
    return X,y,split


def _accuracy(pred: Sequence[int], actual: Sequence[int]) -> float:
    return sum(int(a==b) for a,b in zip(pred,actual))/max(1,len(actual))


def _majority(actual: Sequence[int]) -> float:
    if not actual: return 1.0
    return max(actual.count(c) for c in set(actual))/len(actual)


def train_model(X: List[List[float]], y: List[int], split: List[int], part_id: str,
                device_request: str, epochs: int, seed: int):
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise RuntimeError("Training unavailable: install local PyTorch; live runtime has no PyTorch requirement.") from e
    torch.set_num_threads(1)
    if device_request not in ("cpu", "cuda"): raise ValueError("device must be cpu or cuda")
    if device_request=="cuda" and not torch.cuda.is_available(): raise ValueError("CUDA requested but unavailable; rerun with --device cpu.")
    spec=MODEL_SPECS[part_id]; in_dim=len(FEATURE_NAMES[part_id])
    train=[X[i] for i,s in enumerate(split) if s==0]
    mean=[sum(r[j] for r in train)/len(train) for j in range(in_dim)]
    std=[max(math.sqrt(sum((r[j]-mean[j])**2 for r in train)/len(train)),1e-6) for j in range(in_dim)]
    Z=[[max(-8.,min(8.,(row[j]-mean[j])/std[j])) for j in range(in_dim)] for row in X]
    def run(device_name: str):
        torch.manual_seed(seed); random.seed(seed); torch.use_deterministic_algorithms(True,warn_only=True)
        device=torch.device(device_name)
        modules=[]; linear_ids=[]; idx=0
        for layer_no,(out_dim,act) in enumerate(zip(spec["dims"][1:],spec["acts"])):
            modules.append(nn.Linear(spec["dims"][layer_no],out_dim)); linear_ids.append(str(idx)); idx+=1
            if act=="relu": modules.append(nn.ReLU()); idx+=1
            elif act=="tanh": modules.append(nn.Tanh()); idx+=1
        model=nn.Sequential(*modules).to(device=device,dtype=torch.float32)
        train_ids=[i for i,s in enumerate(split) if s==0]; val_ids=[i for i,s in enumerate(split) if s==1]; test_ids=[i for i,s in enumerate(split) if s==2]
        tx=torch.tensor([Z[i] for i in train_ids],dtype=torch.float32); ty=torch.tensor([y[i] for i in train_ids],dtype=torch.float32 if spec["kind"]=="sigmoid" else torch.long)
        vx=torch.tensor([Z[i] for i in val_ids],dtype=torch.float32,device=device); vy=torch.tensor([y[i] for i in val_ids],dtype=torch.float32 if spec["kind"]=="sigmoid" else torch.long,device=device)
        loader=DataLoader(TensorDataset(tx,ty),batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        loss_fn=nn.BCEWithLogitsLoss() if spec["kind"]=="sigmoid" else nn.CrossEntropyLoss()
        best=float("inf"); best_state=None; patience=0
        for _ in range(epochs):
            model.train()
            for bx,by in loader:
                bx,by=bx.to(device),by.to(device)
                opt.zero_grad(set_to_none=True); logits=model(bx)
                loss=loss_fn(logits.squeeze(-1),by) if spec["kind"]=="sigmoid" else loss_fn(logits,by)
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl=model(vx); val_loss=loss_fn(vl.squeeze(-1),vy) if spec["kind"]=="sigmoid" else loss_fn(vl,vy)
                vloss=float(val_loss.item())
            if vloss<best-1e-5:
                best=vloss; patience=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                patience+=1
                if patience>=7: break
        if best_state is None: raise RuntimeError("no validation checkpoint")
        model.load_state_dict(best_state); model.to(device); model.eval()
        test_ids=[i for i,s in enumerate(split) if s==2]
        def predict(ids):
            xx=torch.tensor([Z[i] for i in ids],dtype=torch.float32,device=device)
            with torch.no_grad(): out=model(xx)
            if spec["kind"]=="sigmoid": return (torch.sigmoid(out.squeeze(-1))>=.5).long().cpu().tolist()
            return out.argmax(dim=1).cpu().tolist()
        val_actual=[y[i] for i in val_ids]; test_actual=[y[i] for i in test_ids]
        vp=predict(val_ids); tp=predict(test_ids)
        metrics={"validation_accuracy":_accuracy(vp,val_actual),"validation_majority_accuracy":_majority(val_actual),
                 "test_accuracy":_accuracy(tp,test_actual),"test_majority_accuracy":_majority(test_actual),"pytorch_version":str(torch.__version__)}
        state={idx:{"weights":model.state_dict()[idx+".weight"].detach().cpu().tolist(),"bias":model.state_dict()[idx+".bias"].detach().cpu().tolist()} for idx in linear_ids}
        return state,mean,std,metrics,device_name
    try:
        state,mean,std,metrics,device=run(device_request)
    except RuntimeError as e:
        oom="out of memory" in str(e).lower() or ("cuda" in str(e).lower() and "memory" in str(e).lower())
        if device_request!="cuda" or not oom: raise
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        state,mean,std,metrics,_=run("cpu"); device="cpu_fallback_after_cuda_oom"
    return state,{"mean":mean,"std":std,**metrics},device


def _sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def _artifact(identity, part_id, state, stats, counts, horizon, neutral_bps, device, dataset_hash, last_ts):
    if stats["validation_accuracy"] < stats["validation_majority_accuracy"]+.02 or stats["test_accuracy"] < stats["test_majority_accuracy"]+.02:
        raise ValueError("Training unavailable: validation and test must each beat majority-label baseline by ≥2 percentage points; no artifact written.")
    spec=MODEL_SPECS[part_id]; layers=[]
    for i,act in enumerate(spec["acts"]):
        layer=state[str(i*2)] if i*2 in [int(k) for k in state] else None
        # Linear-module state indexes differ only by the activation modules preceding them.
        if layer is None:
            keys=sorted(state,key=int); layer=state[keys[i]]
        layers.append({"weights":layer["weights"],"bias":layer["bias"],"activation":act})
    contract={"task":spec["task"],"dims":list(spec["dims"]),"activations":list(spec["acts"]),"labels":list(spec["labels"]),"kind":spec["kind"]}
    return {"format":FORMAT,"format_version":FORMAT_VERSION,"feature_schema":FEATURE_SCHEMA,
        "part_id":part_id,"feature_names":list(FEATURE_NAMES[part_id]),"identity":identity,
        "model_contract":contract,"normalization":{"mean":stats["mean"],"std":stats["std"]},"layers":layers,
        "training":{"status":"validated","approved_for_advisory":True,"split":"chronological_purged",
        "model_version":"3","train_samples":counts[0],"validation_samples":counts[1],"test_samples":counts[2],
        "horizon_bars":horizon,"neutral_threshold_bps":neutral_bps,"dataset_sha256":dataset_hash,
        "dataset_last_timestamp":last_ts,"trainer_version":"3.0","trained_at_utc":datetime.now(timezone.utc).isoformat(),
        "pytorch_version":stats.get("pytorch_version","unknown"),
        "validation_accuracy":stats["validation_accuracy"],"validation_majority_accuracy":stats["validation_majority_accuracy"],
        "test_accuracy":stats["test_accuracy"],"test_majority_accuracy":stats["test_majority_accuracy"],
        "device":device,"dtype":"float32","target_warning":"Part-specific research target; uncalibrated advisory proxy, not strategy PnL, live authorization or profitability evidence"}}


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv",type=Path,help="offline chronological closed-OHLCV CSV; Parts 11/12 also need evidence_json per row")
    ap.add_argument("--part",required=True,choices=FEATURE_NAMES.keys())
    ap.add_argument("--venue",required=True); ap.add_argument("--market-type",required=True)
    ap.add_argument("--instrument-id",required=True); ap.add_argument("--symbol",required=True)
    ap.add_argument("--timeframe",required=True,choices=tuple(_TIMEFRAME_SECONDS))
    ap.add_argument("--horizon",type=int,default=3)
    ap.add_argument("--neutral-bps",type=float,default=20.0)
    ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cpu",help="bounded CPU default; opt-in CUDA retries on CPU after CUDA OOM")
    ap.add_argument("--seed",type=int,default=20260929)
    ap.add_argument("--artifact-dir",type=Path,default=Path(os.environ.get("JARVIS_NEURAL_ARTIFACT_DIR","models/advisory")))
    ap.add_argument("--replace",action="store_true")
    args=ap.parse_args()
    if not 1<=args.horizon<=24 or not 0<=args.neutral_bps<=1000 or not 1<=args.epochs<=200: ap.error("horizon 1-24, neutral-bps 0-1000 and epochs 1-200 required")
    try:
        identity=_identity_arg(args); digest=_sha256_file(args.csv)
        rows=read_closed_csv(args.csv,identity,args.part); last=parse_timestamp(rows[-1]["timestamp"])
        X,y,split=make_samples(rows,args.part,args.horizon,args.neutral_bps); counts=[sum(s==k for s in split) for k in range(3)]
        state,stats,device=train_model(X,y,split,args.part,args.device,args.epochs,args.seed)
        if _sha256_file(args.csv)!=digest: raise ValueError("input CSV changed during training; discard and rerun")
        artifact=_artifact(identity,args.part,state,stats,counts,args.horizon,args.neutral_bps,device,digest,last)
        target=artifact_path(identity,args.part,str(args.artifact_dir))
        if target.exists() and not args.replace: raise ValueError("exact-identity artifact exists; pass --replace explicitly")
        target.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix=".jarvis-model-",suffix=".json",dir=target.parent)
        try:
            with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(artifact,f,separators=(",",":"),allow_nan=False); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,target)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        print(json.dumps({"status":"artifact_created","path":str(target),"identity":identity,"part":args.part,
            "task":MODEL_SPECS[args.part]["task"],"model_contract":artifact["model_contract"],"samples":counts,"device":device,
            "metrics":{k:stats[k] for k in ("validation_accuracy","validation_majority_accuracy","test_accuracy","test_majority_accuracy")},
            "warning":"diagnostic-only task proxy; no calibration/PnL/profitability claim"},sort_keys=True))
        return 0
    except Exception as e:
        print(f"TRAINING UNAVAILABLE: {e}",file=sys.stderr); return 2

if __name__=="__main__": raise SystemExit(main())
