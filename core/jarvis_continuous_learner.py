"""DISABLED. The previous daemon trained XGBoost on np.random vote features every
few minutes (pure noise) and the output was used as a live veto. Models are now
trained OFFLINE only: see train_xgb_meta.py. Runtime consumption is in
jarvis_xgb_meta.py (validated, per-symbol, veto-only, fail-neutral)."""


class ContinuousLearner:
    def __init__(self, *a, **k):
        raise RuntimeError("Live continuous learning is disabled; use train_xgb_meta.py offline")


if __name__ == "__main__":
    raise SystemExit("Disabled: use `python train_xgb_meta.py --symbol BTCUSDT` (offline)")
