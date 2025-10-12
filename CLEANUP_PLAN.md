# Project Cleanup & Reorganization Plan

**Problem**: Messy file structure with duplicates

---

## Current Mess:

```
❌ Multiple copies of same files
❌ hmm_detector.py AND hmm_detector_v2.py
❌ src/regime/ AND user_data/strategies/regime/
❌ Confusion about which file is "source of truth"
```

---

## Clean Structure:

### Option 1: Single Repository (Recommended)

```
C:\kian_trade\
├── pragma-trading-bot/              # Main project
│   ├── src/
│   │   └── regime/
│   │       ├── __init__.py
│   │       └── hmm_detector.py      # ONE file only!
│   │
│   ├── strategies/                   # Freqtrade strategies here
│   │   └── regime_adaptive_strategy.py
│   │
│   ├── tests/
│   ├── docs/
│   ├── user_data/                    # Freqtrade user_data
│   │   ├── data/
│   │   └── backtest_results/
│   │
│   └── freqtrade_config.json         # Config here
│
└── freqtrade/                        # Only Freqtrade installation
    └── (Freqtrade core files)
```

### Option 2: Separate but Linked

```
C:\kian_trade\
├── pragma-trading-bot/              # Source code
│   ├── src/regime/
│   │   └── hmm_detector.py
│   └── strategies/
│       └── regime_adaptive_strategy.py
│
└── freqtrade/                       # Freqtrade
    └── user_data/strategies/
        ├── regime_adaptive_strategy.py  # Symlink
        └── regime/                      # Symlink
```

---

## Cleanup Steps:

### Step 1: Decide on ONE hmm_detector

**Keep**: hmm_detector_v2.py (has best practices)
**Delete**: hmm_detector.py (old version)
**Rename**: hmm_detector_v2.py → hmm_detector.py

### Step 2: Remove Duplicates

**Keep in**: `src/regime/` (source of truth)
**Delete**: `user_data/strategies/regime/` (copy)

### Step 3: Update Strategy Import

```python
# In regime_adaptive_strategy.py:
# Change from:
from regime.hmm_detector import RegimeDetector

# To:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.regime.hmm_detector import EnhancedRegimeDetector
```

### Step 4: Deploy to Freqtrade

Create deployment script:
```python
# scripts/deploy_strategy.py
import shutil
from pathlib import Path

source = Path('src/')
strategy = Path('strategies/regime_adaptive_strategy.py')
target = Path('C:/kian_trade/freqtrade/user_data/strategies/')

# Copy strategy
shutil.copy(strategy, target)

# Copy regime module
shutil.copytree(source / 'regime', target / 'regime', dirs_exist_ok=True)

print("✅ Strategy deployed to Freqtrade!")
```

---

## Action Plan (15 minutes):

1. ✅ Rename hmm_detector_v2.py → hmm_detector.py (replace old)
2. ✅ Delete old hmm_detector.py
3. ✅ Delete user_data/strategies/ (it's a copy)
4. ✅ Update imports in strategy
5. ✅ Create deploy script
6. ✅ Test deployment

---

## Final Clean Structure:

```
pragma-trading-bot/
├── src/
│   └── regime/
│       ├── __init__.py
│       └── hmm_detector.py          # ✅ Enhanced version
│
├── strategies/
│   └── regime_adaptive_strategy.py  # ✅ Uses src/regime
│
├── scripts/
│   └── deploy_to_freqtrade.py       # ✅ Deployment script
│
├── tests/
├── docs/
└── configs/
```

**Freqtrade** only gets deployed files, not development files.

---

## Benefits:

✅ Clear separation: development vs deployment
✅ Single source of truth
✅ No confusion
✅ Easy to update
✅ Professional structure
