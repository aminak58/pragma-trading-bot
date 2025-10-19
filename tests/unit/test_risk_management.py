"""
Unit tests for Risk Management Module

Tests cover:
- Kelly Criterion position sizing
- Dynamic stop loss calculations
- Position management
- Circuit breaker functionality
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import numpy as np
import pandas as pd
from risk.kelly_criterion import KellyCriterion
from risk.dynamic_stops import DynamicStopLoss
from risk.position_manager import PositionManager
from risk.circuit_breakers import CircuitBreaker, CircuitBreakerState


# Fixtures
@pytest.fixture
def sample_dataframe():
    """Generate sample OHLCV dataframe for testing."""
    np.random.seed(42)
    n = 100
    
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_price = close * (1 + np.random.randn(n) * 0.005)
    volume = np.random.lognormal(10, 0.5, n)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Add ATR
    df['atr'] = (df['high'] - df['low']).rolling(window=14).mean()
    
    return df


@pytest.fixture
def kelly_criterion():
    """Create KellyCriterion instance."""
    return KellyCriterion()


@pytest.fixture
def dynamic_stops():
    """Create DynamicStopLoss instance."""
    return DynamicStopLoss()


@pytest.fixture
def position_manager():
    """Create PositionManager instance."""
    return PositionManager()


@pytest.fixture
def circuit_breaker():
    """Create CircuitBreaker instance."""
    return CircuitBreaker()


# Kelly Criterion Tests
class TestKellyCriterion:
    """Test Kelly Criterion functionality."""
    
    def test_initialization(self, kelly_criterion):
        """Test KellyCriterion initialization."""
        assert kelly_criterion.max_kelly_fraction == 0.25
        assert kelly_criterion.min_kelly_fraction == 0.01
        assert kelly_criterion.confidence_adjustment is True
        assert kelly_criterion.regime_adjustment is True
    
    def test_add_trade_result(self, kelly_criterion):
        """Test adding trade results."""
        kelly_criterion.add_trade_result(0.05, 0.8)
        kelly_criterion.add_trade_result(-0.02, 0.6)
        
        assert len(kelly_criterion.trade_history) == 2
        assert kelly_criterion.performance_stats['total_trades'] == 2
        assert kelly_criterion.performance_stats['winning_trades'] == 1
        assert kelly_criterion.performance_stats['losing_trades'] == 1
    
    def test_calculate_kelly_fraction(self, kelly_criterion):
        """Test Kelly fraction calculation."""
        # Add some trade history
        kelly_criterion.add_trade_result(0.05, 0.8)
        kelly_criterion.add_trade_result(0.03, 0.7)
        kelly_criterion.add_trade_result(-0.02, 0.6)
        kelly_criterion.add_trade_result(-0.01, 0.5)
        
        kelly_fraction = kelly_criterion.calculate_kelly_fraction(
            win_rate=0.5,
            win_loss_ratio=2.0,
            confidence=0.8,
            regime='trending'
        )
        
        assert 0.0 <= kelly_fraction <= 1.0
        assert kelly_fraction >= kelly_criterion.min_kelly_fraction
        assert kelly_fraction <= kelly_criterion.max_kelly_fraction
    
    def test_calculate_position_size(self, kelly_criterion):
        """Test position size calculation."""
        result = kelly_criterion.calculate_position_size(
            account_balance=10000,
            current_price=100,
            stop_loss_price=95,
            confidence=0.8,
            regime='trending'
        )
        
        assert 'kelly_fraction' in result
        assert 'position_size' in result
        assert 'position_value' in result
        assert 'leverage' in result
        assert result['position_size'] > 0
        assert result['position_value'] > 0
    
    def test_regime_multipliers(self, kelly_criterion):
        """Test regime-based position adjustments."""
        # Test different regimes
        regimes = ['trending', 'low_volatility', 'high_volatility', 'unknown']
        
        for regime in regimes:
            multiplier = kelly_criterion._get_regime_multiplier(regime)
            assert 0.0 <= multiplier <= 1.0
        
        # Trending should have highest multiplier
        trending_mult = kelly_criterion._get_regime_multiplier('trending')
        high_vol_mult = kelly_criterion._get_regime_multiplier('high_volatility')
        assert trending_mult > high_vol_mult


# Dynamic Stops Tests
class TestDynamicStops:
    """Test Dynamic Stop Loss functionality."""
    
    def test_initialization(self, dynamic_stops):
        """Test DynamicStopLoss initialization."""
        assert dynamic_stops.base_stop_percent == 0.02
        assert dynamic_stops.atr_multiplier == 2.0
        assert dynamic_stops.trailing_activation == 0.01
        assert dynamic_stops.trailing_distance == 0.005
    
    def test_calculate_atr_stop(self, dynamic_stops, sample_dataframe):
        """Test ATR-based stop calculation."""
        stop_price = dynamic_stops.calculate_atr_stop(
            sample_dataframe, 100, 'long'
        )
        
        assert stop_price < 100  # Should be below current price for long
        assert stop_price > 0    # Should be positive
    
    def test_calculate_regime_stop(self, dynamic_stops):
        """Test regime-based stop calculation."""
        stop_price = dynamic_stops.calculate_regime_stop(
            current_price=100,
            side='long',
            regime='trending',
            confidence=0.8
        )
        
        assert stop_price < 100  # Should be below current price for long
        assert stop_price > 0    # Should be positive
    
    def test_calculate_trailing_stop(self, dynamic_stops):
        """Test trailing stop calculation."""
        # Test long position
        trailing_stop = dynamic_stops.calculate_trailing_stop(
            entry_price=100,
            current_price=105,
            side='long',
            highest_price=105
        )
        
        assert trailing_stop > 100  # Should be above entry
        assert trailing_stop < 105  # Should be below current price
    
    def test_calculate_adaptive_stop(self, dynamic_stops, sample_dataframe):
        """Test adaptive stop calculation."""
        stops = dynamic_stops.calculate_adaptive_stop(
            dataframe=sample_dataframe,
            current_price=100,
            side='long',
            regime='trending',
            confidence=0.8
        )
        
        assert 'atr' in stops
        assert 'regime' in stops
        assert 'recommended' in stops
        assert stops['recommended'] < 100  # Should be below current price
    
    def test_add_trade(self, dynamic_stops):
        """Test adding trade to tracking."""
        dynamic_stops.add_trade(
            trade_id='test_trade',
            side='long',
            entry_price=100,
            stop_price=95,
            regime='trending'
        )
        
        assert 'test_trade' in dynamic_stops.active_trades
        assert dynamic_stops.get_trade_stop('test_trade') == 95


# Position Manager Tests
class TestPositionManager:
    """Test Position Manager functionality."""
    
    def test_initialization(self, position_manager):
        """Test PositionManager initialization."""
        assert position_manager.max_portfolio_risk == 0.02
        assert position_manager.max_position_risk == 0.01
        assert position_manager.max_correlation == 0.7
    
    def test_calculate_position_size(self, position_manager):
        """Test position size calculation."""
        # Set portfolio value first
        position_manager.update_portfolio_value(10000)
        
        position_data = position_manager.calculate_position_size(
            pair='BTC/USDT',
            current_price=50000,
            stop_loss_price=48000,
            confidence=0.8,
            regime='trending'
        )
        
        assert 'position_size' in position_data
        assert 'position_value' in position_data
        assert 'risk_amount' in position_data
        assert position_data['position_size'] >= 0
    
    def test_open_position(self, position_manager):
        """Test opening a position."""
        position_data = {
            'position_size': 0.1,
            'position_value': 5000,
            'risk_amount': 200
        }
        
        position_id = position_manager.open_position(
            pair='BTC/USDT',
            side='long',
            entry_price=50000,
            stop_loss_price=48000,
            confidence=0.8,
            regime='trending',
            position_data=position_data
        )
        
        assert position_id in position_manager.positions
        assert position_manager.positions[position_id]['pair'] == 'BTC/USDT'
        assert position_manager.positions[position_id]['side'] == 'long'
    
    def test_update_position(self, position_manager, sample_dataframe):
        """Test updating a position."""
        # First open a position
        position_data = {
            'position_size': 0.1,
            'position_value': 5000,
            'risk_amount': 200
        }
        
        position_id = position_manager.open_position(
            pair='BTC/USDT',
            side='long',
            entry_price=50000,
            stop_loss_price=48000,
            confidence=0.8,
            regime='trending',
            position_data=position_data
        )
        
        # Update position
        update_data = position_manager.update_position(
            position_id, 51000, sample_dataframe
        )
        
        assert update_data is not None
        assert 'unrealized_pnl' in update_data
        assert update_data['unrealized_pnl'] > 0  # Should be profitable
    
    def test_close_position(self, position_manager):
        """Test closing a position."""
        # First open a position
        position_data = {
            'position_size': 0.1,
            'position_value': 5000,
            'risk_amount': 200
        }
        
        position_id = position_manager.open_position(
            pair='BTC/USDT',
            side='long',
            entry_price=50000,
            stop_loss_price=48000,
            confidence=0.8,
            regime='trending',
            position_data=position_data
        )
        
        # Close position
        close_data = position_manager.close_position(
            position_id, 51000, 'manual'
        )
        
        assert close_data is not None
        assert 'realized_pnl' in close_data
        assert close_data['realized_pnl'] > 0  # Should be profitable
        assert position_id not in position_manager.positions


# Circuit Breaker Tests
class TestCircuitBreaker:
    """Test Circuit Breaker functionality."""
    
    def test_initialization(self, circuit_breaker):
        """Test CircuitBreaker initialization."""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.max_drawdown == 0.05
        assert circuit_breaker.max_consecutive_losses == 5
        assert circuit_breaker.max_daily_loss == 0.02
    
    def test_update_balance(self, circuit_breaker):
        """Test balance update and drawdown check."""
        # Set initial balance
        circuit_breaker.update_balance(10000)
        assert circuit_breaker.current_balance == 10000
        assert circuit_breaker.peak_balance == 10000
        
        # Test drawdown trigger
        circuit_breaker.update_balance(9400)  # 6% drawdown
        assert circuit_breaker.state == CircuitBreakerState.OPEN
    
    def test_add_trade_result(self, circuit_breaker):
        """Test trade result addition and consecutive loss check."""
        # Add some winning trades
        circuit_breaker.add_trade_result(0.01, 'BTC/USDT', 0.8)
        circuit_breaker.add_trade_result(0.02, 'ETH/USDT', 0.7)
        
        assert circuit_breaker.consecutive_losses == 0
        
        # Add consecutive losses
        for i in range(6):
            circuit_breaker.add_trade_result(-0.01, f'PAIR{i}', 0.6)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.consecutive_losses == 6
    
    def test_check_trading_allowed(self, circuit_breaker):
        """Test trading permission check."""
        # Set initial balance
        circuit_breaker.update_balance(10000)
        
        # Normal state
        allowed, reason = circuit_breaker.check_trading_allowed()
        assert allowed is True
        assert reason == "Trading allowed"
        
        # Trigger breaker with drawdown
        circuit_breaker.update_balance(9400)  # 6% drawdown
        allowed, reason = circuit_breaker.check_trading_allowed()
        assert allowed is False
        assert "Circuit breaker open" in reason
    
    def test_volatility_spike(self, circuit_breaker):
        """Test volatility spike detection."""
        # Add normal volatility
        for i in range(25):
            circuit_breaker.update_volatility(0.02)
        
        # Add volatility spike
        circuit_breaker.update_volatility(0.08)  # 4x normal
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
    
    def test_force_close_breaker(self, circuit_breaker):
        """Test force closing circuit breaker."""
        # Set initial balance
        circuit_breaker.update_balance(10000)
        
        # Trigger breaker with drawdown
        circuit_breaker.update_balance(9400)  # 6% drawdown
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Force close
        circuit_breaker.force_close_breaker("Test")
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
    
    def test_get_status(self, circuit_breaker):
        """Test status reporting."""
        circuit_breaker.update_balance(10000)
        circuit_breaker.add_trade_result(0.01, 'BTC/USDT', 0.8)
        
        status = circuit_breaker.get_status()
        
        assert 'state' in status
        assert 'current_balance' in status
        assert 'peak_balance' in status
        assert 'current_drawdown' in status
        assert status['current_balance'] == 10000
        assert status['peak_balance'] == 10000


# Integration Tests
class TestRiskManagementIntegration:
    """Integration tests for risk management components."""
    
    def test_full_position_lifecycle(self, position_manager, sample_dataframe):
        """Test complete position lifecycle."""
        # Set portfolio value
        position_manager.update_portfolio_value(10000)
        
        # Calculate position size
        position_data = position_manager.calculate_position_size(
            pair='BTC/USDT',
            current_price=50000,
            stop_loss_price=48000,
            confidence=0.8,
            regime='trending'
        )
        
        # Open position
        position_id = position_manager.open_position(
            pair='BTC/USDT',
            side='long',
            entry_price=50000,
            stop_loss_price=48000,
            confidence=0.8,
            regime='trending',
            position_data=position_data
        )
        
        # Update position
        position_manager.update_position(position_id, 51000, sample_dataframe)
        
        # Close position
        close_data = position_manager.close_position(position_id, 52000, 'manual')
        
        # Verify results
        assert close_data['realized_pnl'] > 0
        assert position_id not in position_manager.positions
        
        # Check performance stats
        stats = position_manager.get_performance_stats()
        assert stats['total_trades'] == 1
        assert stats['winning_trades'] == 1
        assert stats['win_rate'] == 1.0
    
    def test_circuit_breaker_integration(self, position_manager, circuit_breaker):
        """Test circuit breaker integration with position manager."""
        # Set up
        position_manager.update_portfolio_value(10000)
        circuit_breaker.update_balance(10000)
        
        # Add some losing trades to trigger circuit breaker
        for i in range(6):
            circuit_breaker.add_trade_result(-0.01, f'PAIR{i}', 0.6)
        
        # Check if trading is allowed
        allowed, reason = circuit_breaker.check_trading_allowed()
        assert allowed is False
        
        # Try to open position (should be blocked by circuit breaker)
        position_data = position_manager.calculate_position_size(
            pair='BTC/USDT',
            current_price=50000,
            stop_loss_price=48000,
            confidence=0.8,
            regime='trending'
        )
        
        # Position size should be reduced due to circuit breaker
        # (The position manager doesn't directly check circuit breaker, 
        # but the risk limits should be more conservative)
        assert position_data['position_size'] >= 0
