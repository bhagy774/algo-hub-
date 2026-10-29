# Jarvis Clean Architecture

Welcome to the organized and modular Jarvis Trading Engine repository.

## Directory Structure

```text
jarvis_clean/
├── core/         # Core Live Trading Systems & Algorithms
│   ├── jarvis_FIXED.py               # Main Live Trading Engine
│   ├── jarvis_strategy_approval.py   # Multi-part Strategy Gate & Consensus
│   ├── jarvis_safe_entry.py          # Dynamic ATR TPSL & Wall Snapping
│   ├── jarvis_multicoin_pipeline.py  # Multi-symbol Execution Pipeline
│   ├── quantitative_math.py          # Math, Horizon Roles & Signal Transforms
│   ├── binance_data.py               # High-frequency Candle & Stream Feeds
│   ├── mode_engine.py                # Regime Detection & Directional Consensus
│   ├── part1_FIXED.py ... part12_FIXED.py  # Algorithmic Trading Parts (1 to 12)
│   └── ...
├── tests/        # Comprehensive Automated Test Suite
│   ├── test_multicoin_pipeline.py
│   ├── test_strategy_approval.py
│   ├── test_mode_parts_integration.py
│   └── ... (All unit, regression, and scenario tests)
├── training/     # Machine Learning & AI Model Training
│   ├── train_xgb_meta.py
│   ├── train_xgb_parts.py
│   ├── colab_mega_trainer.py
│   └── run_local_mega_training.py
├── tools/        # Audits, Diagnostics & Utilities
│   ├── audit_bugs.py
│   ├── check_regime.py
│   ├── fix_tests.py
│   └── ...
├── models/       # Production XGBoost & Neural Weights
│   ├── xgb_mode_BTCUSDT.json
│   ├── xgb_mode_ETHUSDT.json
│   └── ...
├── pytest.ini    # Test runner configuration (pythonpath = core .)
├── requirements.txt
└── .gitignore
```

## Running Tests

From the repository root:
```bash
pytest tests/
```

## Running Live Engine

From the repository root:
```bash
python -m core.jarvis_FIXED
```
