#!/usr/bin/env python3
"""
JARVIS 100x Colab Brain Importer & Exporter for Windows PowerShell
==================================================================
1. To export your local JARVIS code into a single ZIP to upload to Google Colab:
       python import_colab_brains.py --pack-for-colab

2. Once `colab_mega_trainer.py` finishes on Google Colab and downloads `jarvis_trained_brains.zip`,
   install all trained Neural Brains, XGBoost Models, and Pre-Trained Self-Learning Memory into
   `C:\\jarvis` and `C:\\jarvis_out` with 1 command:
       python import_colab_brains.py jarvis_trained_brains.zip
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = Path(r"C:\jarvis_out") if Path(r"C:\jarvis_out").is_dir() else ROOT

COLAB_CODE_FILES = (
    "colab_mega_trainer.py",
    "quantitative_math.py",
    "neural_advisory.py",
    "train_part_advisory.py",
    "train_xgb_meta.py",
    "train_xgb_parts.py",
    "jarvis_xgb_meta.py",
    "jarvis_xgb_mode.py",
    "jarvis_xgboost_engine.py",
    "jarvis_part_snapshot.py",
    "jarvis_self_learning.py",
    "jarvis_strategy_approval.py",
    "binance_timeframes.py",
)


def pack_for_colab(out_zip: Path) -> int:
    """Create `jarvis_colab_upload.zip` containing only the lightweight Python files needed on Colab."""
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in COLAB_CODE_FILES:
            fpath = ROOT / fname
            if fpath.is_file():
                zf.write(fpath, fname)
        np_dir = ROOT / "neural_parts"
        if np_dir.is_dir():
            for p in np_dir.rglob("*.py"):
                zf.write(p, str(p.relative_to(ROOT)).replace("\\", "/"))
    print("=================================================================")
    print(f"[OK] Created Colab Upload Bundle: {out_zip} ({out_zip.stat().st_size / 1024:.1f} KB)")
    print("=================================================================")
    print("Google Colab Pro Steps:")
    print("  1. Open https://colab.research.google.com/ -> New Notebook")
    print("  2. Runtime -> Change runtime type -> Select GPU (A100 / L4 / T4)")
    print("  3. Upload `jarvis_colab_upload.zip` to Colab files pane (left sidebar)")
    print("  4. Run this single cell in Colab:")
    print("       !unzip -o jarvis_colab_upload.zip && pip install -q xgboost scikit-learn requests pandas numpy torch && python colab_mega_trainer.py")
    print("  5. When `jarvis_trained_brains.zip` downloads, put it in C:\\jarvis and run:")
    print("       C:\\jarvis\\jenv\\Scripts\\python.exe C:\\jarvis\\import_colab_brains.py jarvis_trained_brains.zip")
    return 0


def import_trained_zip(zip_path: Path) -> int:
    if not zip_path.is_file():
        print(f"[ERROR] File not found: {zip_path}", file=sys.stderr)
        return 2

    models_dest = ROOT / "models"
    models_dest.mkdir(parents=True, exist_ok=True)
    installed_models = 0
    self_learn_merged = False

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            norm = member.replace("\\", "/").lstrip("/")
            if not norm or norm.endswith("/"):
                continue
            if norm.startswith("models/"):
                target = ROOT / norm
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                installed_models += 1
            elif norm == "self_learning_state.json":
                colab_state = json.loads(zf.read(member).decode("utf-8"))
                local_state_path = OUT_DIR / "self_learning_state.json"
                if local_state_path.is_file():
                    try:
                        local_state = json.loads(local_state_path.read_text(encoding="utf-8"))
                        # Preserve local live trade_memory and open_trade_snapshots while adopting pre-trained weights & stats
                        colab_mem = list(colab_state.get("trade_memory") or [])
                        local_mem = list(local_state.get("trade_memory") or [])
                        colab_state["trade_memory"] = (colab_mem + local_mem)[-250:]
                        colab_state["open_trade_snapshots"] = local_state.get("open_trade_snapshots") or {}
                        colab_state["trade_lessons"] = max(
                            int(colab_state.get("trade_lessons", 0)),
                            int(local_state.get("trade_lessons", 0)),
                        )
                    except Exception:
                        pass
                local_state_path.parent.mkdir(parents=True, exist_ok=True)
                local_state_path.write_text(json.dumps(colab_state, indent=2), encoding="utf-8")
                self_learn_merged = True

    print("=================================================================")
    print(f"[OK] Successfully Imported Colab Trained Brains from: {zip_path.name}")
    print(f"   - Installed Model Files in C:\\jarvis\\models : {installed_models}")
    print(f"   - Pre-Trained Self-Learning Memory Updated  : {'YES' if self_learn_merged else 'NO'}")
    print("=================================================================")
    manifest_path = models_dest / "colab_training_manifest.json"
    if manifest_path.is_file():
        print("Training Manifest Summary:")
        print(manifest_path.read_text(encoding="utf-8"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip_file", nargs="?", help="Path to jarvis_trained_brains.zip downloaded from Google Colab")
    ap.add_argument("--pack-for-colab", action="store_true", help="Create jarvis_colab_upload.zip to upload to Colab")
    ap.add_argument("--out-upload", default=str(ROOT / "jarvis_colab_upload.zip"))
    args = ap.parse_args()

    if args.pack_for_colab or not args.zip_file:
        return pack_for_colab(Path(args.out_upload))
    return import_trained_zip(Path(args.zip_file))


if __name__ == "__main__":
    raise SystemExit(main())
