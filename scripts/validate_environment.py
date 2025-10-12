"""
Environment Validation Script for Pragma Trading Bot
Validates that all required dependencies are installed and working
"""

import sys


def validate_python_version():
    """Validate Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor == 11:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} (Expected 3.11)")
        return False


def validate_imports():
    """Validate all required imports"""
    print("\n📦 Checking required packages...")
    
    packages = [
        ("freqtrade", "Freqtrade core"),
        ("freqtrade.freqai.data_kitchen", "FreqAI"),
        ("hmmlearn.hmm", "HMM Learn"),
        ("sklearn", "Scikit-learn"),
        ("xgboost", "XGBoost"),
        ("catboost", "CatBoost"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("talib", "TA-Lib"),
        ("requests", "Requests"),
        ("pytest", "Pytest"),
    ]
    
    all_success = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            all_success = False
    
    return all_success


def validate_freqtrade():
    """Validate Freqtrade installation"""
    print("\n🤖 Checking Freqtrade...")
    
    try:
        import freqtrade
        from freqtrade import __version__
        print(f"   ✅ Freqtrade version: {__version__}")
        return True
    except Exception as e:
        print(f"   ❌ Freqtrade error: {e}")
        return False


def validate_freqai():
    """Validate FreqAI components"""
    print("\n🧠 Checking FreqAI...")
    
    try:
        from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
        from freqtrade.freqai.prediction_models.CatboostClassifier import (
            CatboostClassifier
        )
        print("   ✅ FreqAI DataKitchen")
        print("   ✅ FreqAI CatboostClassifier")
        return True
    except Exception as e:
        print(f"   ❌ FreqAI error: {e}")
        return False


def validate_hmm():
    """Validate HMM components"""
    print("\n📊 Checking HMM (hmmlearn)...")
    
    try:
        from hmmlearn.hmm import GaussianHMM
        import numpy as np
        
        # Test basic HMM creation
        model = GaussianHMM(n_components=3, covariance_type="full")
        print("   ✅ GaussianHMM initialized")
        
        # Test basic fit
        X = np.random.randn(100, 4)
        model.fit(X)
        print("   ✅ HMM training works")
        
        return True
    except Exception as e:
        print(f"   ❌ HMM error: {e}")
        return False


def validate_ml_libraries():
    """Validate ML libraries"""
    print("\n🤖 Checking ML libraries...")
    
    try:
        import sklearn
        import xgboost
        import catboost
        
        print(f"   ✅ scikit-learn {sklearn.__version__}")
        print(f"   ✅ XGBoost {xgboost.__version__}")
        print(f"   ✅ CatBoost {catboost.__version__}")
        return True
    except Exception as e:
        print(f"   ❌ ML libraries error: {e}")
        return False


def main():
    """Run all validation checks"""
    print("=" * 60)
    print("🔍 Pragma Trading Bot - Environment Validation")
    print("=" * 60)
    
    results = []
    
    results.append(("Python Version", validate_python_version()))
    results.append(("Package Imports", validate_imports()))
    results.append(("Freqtrade", validate_freqtrade()))
    results.append(("FreqAI", validate_freqai()))
    results.append(("HMM (hmmlearn)", validate_hmm()))
    results.append(("ML Libraries", validate_ml_libraries()))
    
    print("\n" + "=" * 60)
    print("📋 Validation Summary")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All validations passed! Environment is ready.")
        print("=" * 60)
        return 0
    else:
        print("❌ Some validations failed. Please check errors above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
