"""
Tests for execution safety and mode separation

Critical tests to prevent accidental live trading.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import os
from execution import ExecutionMode, SimulatedExecutor, LiveExecutor


@pytest.fixture
def backtest_config():
    """Sample backtest configuration."""
    return {
        'stake_currency': 'USDT',
        'timeframe': '5m',
        'dry_run': True,
        'strategy': 'RegimeAdaptiveStrategy'
    }


@pytest.fixture
def live_config():
    """Sample live configuration (production-like for testing)."""
    return {
        'stake_currency': 'USDT',
        'timeframe': '5m',
        'dry_run': False,
        'strategy': 'RegimeAdaptiveStrategy',
        'exchange': {
            'name': 'binance',
            'key': 'real_api_key_12345',  # No 'test' pattern
            'secret': 'real_secret_67890'  # No 'test' pattern
        }
    }


class TestExecutionMode:
    """Test execution mode enum."""
    
    def test_execution_modes_exist(self):
        """Test that all execution modes are defined."""
        assert hasattr(ExecutionMode, 'BACKTEST')
        assert hasattr(ExecutionMode, 'DRY_RUN')
        assert hasattr(ExecutionMode, 'LIVE')
    
    def test_mode_values(self):
        """Test mode string values."""
        assert ExecutionMode.BACKTEST.value == "backtest"
        assert ExecutionMode.DRY_RUN.value == "dry-run"
        assert ExecutionMode.LIVE.value == "live"


class TestSimulatedExecutor:
    """Test simulated executor (backtest/dry-run)."""
    
    def test_backtest_initialization(self, backtest_config):
        """Test backtest executor initializes correctly."""
        executor = SimulatedExecutor(backtest_config, mode="backtest")
        
        assert executor.mode == ExecutionMode.BACKTEST
        assert executor.config['dry_run'] is True
        assert not executor.can_execute_live_trades()
    
    def test_dry_run_initialization(self, backtest_config):
        """Test dry-run executor initializes correctly."""
        executor = SimulatedExecutor(backtest_config, mode="dry-run")
        
        assert executor.mode == ExecutionMode.DRY_RUN
        assert executor.config['dry_run'] is True
        assert not executor.can_execute_live_trades()
    
    def test_forces_dry_run_true(self):
        """Test that SimulatedExecutor forces dry_run=True."""
        config = {
            'stake_currency': 'USDT',
            'timeframe': '5m',
            'dry_run': False  # Intentionally False
        }
        
        executor = SimulatedExecutor(config, mode="backtest")
        
        # Should be forced to True
        assert executor.config['dry_run'] is True
    
    def test_rejects_live_mode(self, backtest_config):
        """Test that SimulatedExecutor rejects live mode."""
        with pytest.raises(ValueError, match="only supports 'backtest' or 'dry-run'"):
            SimulatedExecutor(backtest_config, mode="live")
    
    def test_execution_context(self, backtest_config):
        """Test execution context information."""
        executor = SimulatedExecutor(backtest_config)
        
        context = executor.get_execution_context()
        
        assert context['executor_type'] == 'SimulatedExecutor'
        assert context['can_execute_live'] is False
        assert context['safe_for_testing'] is True


class TestLiveExecutor:
    """Test live executor (real trading)."""
    
    def test_requires_explicit_confirmation(self, live_config):
        """Test that LiveExecutor requires confirm_live=True."""
        with pytest.raises(ValueError, match="explicit confirmation"):
            LiveExecutor(live_config, confirm_live=False)
    
    def test_requires_environment_variable(self, live_config):
        """Test that LiveExecutor requires PRAGMA_ALLOW_LIVE env var."""
        # Ensure env var is not set
        os.environ.pop('PRAGMA_ALLOW_LIVE', None)
        
        with pytest.raises(RuntimeError, match="not enabled in environment"):
            LiveExecutor(live_config, confirm_live=True)
    
    def test_requires_global_flag(self, live_config):
        """Test that LiveExecutor requires global flag."""
        os.environ['PRAGMA_ALLOW_LIVE'] = 'true'
        
        # Global flag not set
        LiveExecutor.enable_live_trading(False)
        
        with pytest.raises(RuntimeError, match="not globally enabled"):
            LiveExecutor(live_config, confirm_live=True)
        
        # Cleanup
        os.environ.pop('PRAGMA_ALLOW_LIVE', None)
    
    def test_rejects_dry_run_config(self):
        """Test that LiveExecutor rejects dry_run=True config."""
        config = {
            'stake_currency': 'USDT',
            'timeframe': '5m',
            'dry_run': True,  # Should fail
            'exchange': {'name': 'binance', 'key': 'x', 'secret': 'y'}
        }
        
        os.environ['PRAGMA_ALLOW_LIVE'] = 'true'
        LiveExecutor.enable_live_trading(True)
        
        with pytest.raises(ValueError, match="cannot use dry_run config"):
            LiveExecutor(config, confirm_live=True)
        
        # Cleanup
        os.environ.pop('PRAGMA_ALLOW_LIVE', None)
        LiveExecutor.enable_live_trading(False)
    
    def test_rejects_missing_credentials(self):
        """Test that LiveExecutor requires API credentials."""
        config = {
            'stake_currency': 'USDT',
            'timeframe': '5m',
            'dry_run': False,
            'exchange': {'name': 'binance'}  # Missing key/secret
        }
        
        os.environ['PRAGMA_ALLOW_LIVE'] = 'true'
        LiveExecutor.enable_live_trading(True)
        
        with pytest.raises(ValueError, match="requires exchange API key"):
            LiveExecutor(config, confirm_live=True)
        
        # Cleanup
        os.environ.pop('PRAGMA_ALLOW_LIVE', None)
        LiveExecutor.enable_live_trading(False)
    
    def test_rejects_example_config(self):
        """Test that LiveExecutor rejects configs with test patterns."""
        config = {
            'stake_currency': 'USDT',
            'timeframe': '5m',
            'dry_run': False,
            'exchange': {
                'name': 'binance',
                'key': 'example_key_12345',  # Contains 'example' - should be rejected
                'secret': 'production_secret'
            }
        }
        
        os.environ['PRAGMA_ALLOW_LIVE'] = 'true'
        LiveExecutor.enable_live_trading(True)
        
        with pytest.raises(ValueError, match="contains test patterns"):
            LiveExecutor(config, confirm_live=True)
        
        # Cleanup
        os.environ.pop('PRAGMA_ALLOW_LIVE', None)
        LiveExecutor.enable_live_trading(False)
    
    def test_can_execute_live_trades(self, live_config):
        """Test that LiveExecutor can execute live trades."""
        os.environ['PRAGMA_ALLOW_LIVE'] = 'true'
        LiveExecutor.enable_live_trading(True)
        
        executor = LiveExecutor(live_config, confirm_live=True)
        
        assert executor.can_execute_live_trades() is True
        
        # Cleanup
        os.environ.pop('PRAGMA_ALLOW_LIVE', None)
        LiveExecutor.enable_live_trading(False)
    
    def test_execution_context(self, live_config):
        """Test live execution context."""
        os.environ['PRAGMA_ALLOW_LIVE'] = 'true'
        LiveExecutor.enable_live_trading(True)
        
        executor = LiveExecutor(live_config, confirm_live=True)
        context = executor.get_execution_context()
        
        assert context['executor_type'] == 'LiveExecutor'
        assert context['can_execute_live'] is True
        assert context['safe_for_testing'] is False
        assert 'REAL MONEY' in context['warning']
        
        # Cleanup
        os.environ.pop('PRAGMA_ALLOW_LIVE', None)
        LiveExecutor.enable_live_trading(False)


class TestSafetyMechanisms:
    """Test overall safety mechanisms."""
    
    def test_simulated_cannot_become_live(self, backtest_config):
        """Test that SimulatedExecutor cannot be switched to live."""
        executor = SimulatedExecutor(backtest_config)
        
        # Should not be possible to change mode
        assert executor.mode == ExecutionMode.BACKTEST
        
        # Verify it cannot execute live
        assert not executor.can_execute_live_trades()
    
    def test_multiple_safety_layers(self, live_config):
        """Test that live trading requires multiple confirmations."""
        # Layer 1: confirm_live must be True
        # Layer 2: Environment variable must be set
        # Layer 3: Global flag must be enabled
        # Layer 4: Config must be production-ready
        
        # All layers must pass
        os.environ['PRAGMA_ALLOW_LIVE'] = 'true'
        LiveExecutor.enable_live_trading(True)
        
        # This should succeed (all safety layers passed)
        executor = LiveExecutor(live_config, confirm_live=True)
        assert executor.can_execute_live_trades()
        
        # Cleanup
        os.environ.pop('PRAGMA_ALLOW_LIVE', None)
        LiveExecutor.enable_live_trading(False)
    
    def test_safety_phrase_mechanism(self, live_config):
        """Test optional safety phrase."""
        os.environ['PRAGMA_ALLOW_LIVE'] = 'true'
        LiveExecutor.enable_live_trading(True)
        
        # Correct phrase should work
        correct_phrase = "I understand this uses real money"
        executor = LiveExecutor(
            live_config,
            confirm_live=True,
            safety_phrase=correct_phrase
        )
        assert executor.can_execute_live_trades()
        
        # Wrong phrase should fail
        with pytest.raises(ValueError, match="Incorrect safety phrase"):
            LiveExecutor(
                live_config,
                confirm_live=True,
                safety_phrase="wrong phrase"
            )
        
        # Cleanup
        os.environ.pop('PRAGMA_ALLOW_LIVE', None)
        LiveExecutor.enable_live_trading(False)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
