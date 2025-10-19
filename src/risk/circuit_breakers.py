"""
Circuit Breaker System

Implements circuit breakers to protect against:
- Excessive drawdowns
- Rapid consecutive losses
- Market volatility spikes
- System failures
- Risk limit breaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Trading halted
    HALF_OPEN = "half_open"  # Limited trading allowed


class CircuitBreaker:
    """
    Circuit breaker system for risk protection.
    
    Monitors various risk metrics and halts trading when thresholds are breached.
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.05,           # 5% max drawdown
        max_consecutive_losses: int = 5,       # Max consecutive losses
        max_daily_loss: float = 0.02,         # 2% max daily loss
        max_volatility_spike: float = 3.0,    # 3x normal volatility
        cooldown_minutes: int = 60,           # 60 min cooldown
        recovery_threshold: float = 0.01      # 1% recovery to close
    ):
        """
        Initialize circuit breaker.
        
        Args:
            max_drawdown: Maximum allowed drawdown (as fraction)
            max_consecutive_losses: Maximum consecutive losing trades
            max_daily_loss: Maximum daily loss (as fraction)
            max_volatility_spike: Maximum volatility spike multiplier
            cooldown_minutes: Cooldown period in minutes
            recovery_threshold: Recovery threshold to close breaker
        """
        self.max_drawdown = max_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_loss = max_daily_loss
        self.max_volatility_spike = max_volatility_spike
        self.cooldown_minutes = cooldown_minutes
        self.recovery_threshold = recovery_threshold
        
        # State tracking
        self.state = CircuitBreakerState.CLOSED
        self.triggered_at = None
        self.trigger_reason = None
        
        # Performance tracking
        self.peak_balance = 0.0
        self.current_balance = 0.0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trade_history = []
        
        # Volatility tracking
        self.volatility_history = []
        self.normal_volatility = None
        
        # Circuit breaker history
        self.breaker_history = []
        
        logger.info("CircuitBreaker initialized")
    
    def update_balance(self, new_balance: float) -> None:
        """
        Update account balance and check for drawdown.
        
        Args:
            new_balance: Current account balance
        """
        self.current_balance = new_balance
        
        # Update peak balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
        
        # Check for drawdown
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - new_balance) / self.peak_balance
            
            if drawdown > self.max_drawdown:
                self._trigger_breaker(
                    reason=f"Drawdown exceeded: {drawdown:.2%} > {self.max_drawdown:.2%}"
                )
    
    def add_trade_result(
        self,
        pnl: float,
        pair: str,
        confidence: float = 1.0
    ) -> None:
        """
        Add trade result and check for consecutive losses.
        
        Args:
            pnl: Profit/loss of the trade
            pair: Trading pair
            confidence: Trade confidence
        """
        # Add to trade history
        trade_record = {
            'timestamp': pd.Timestamp.now(),
            'pnl': pnl,
            'pair': pair,
            'confidence': confidence
        }
        self.trade_history.append(trade_record)
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Check consecutive loss limit
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._trigger_breaker(
                reason=f"Consecutive losses exceeded: {self.consecutive_losses} >= {self.max_consecutive_losses}"
            )
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss * self.peak_balance:
            self._trigger_breaker(
                reason=f"Daily loss exceeded: {self.daily_pnl:.2f} < {self.max_daily_loss:.2%} of peak"
            )
    
    def update_volatility(self, current_volatility: float) -> None:
        """
        Update volatility and check for spikes.
        
        Args:
            current_volatility: Current market volatility
        """
        # Add to volatility history
        self.volatility_history.append({
            'timestamp': pd.Timestamp.now(),
            'volatility': current_volatility
        })
        
        # Keep only recent history (last 100 observations)
        if len(self.volatility_history) > 100:
            self.volatility_history = self.volatility_history[-100:]
        
        # Calculate normal volatility (rolling average)
        if len(self.volatility_history) >= 20:
            recent_volatilities = [v['volatility'] for v in self.volatility_history[-20:]]
            self.normal_volatility = np.mean(recent_volatilities)
            
            # Check for volatility spike
            if self.normal_volatility > 0:
                volatility_ratio = current_volatility / self.normal_volatility
                
                if volatility_ratio > self.max_volatility_spike:
                    self._trigger_breaker(
                        reason=f"Volatility spike detected: {volatility_ratio:.2f}x normal"
                    )
    
    def check_trading_allowed(self) -> Tuple[bool, str]:
        """
        Check if trading is currently allowed.
        
        Returns:
            Tuple of (allowed, reason)
        """
        if self.state == CircuitBreakerState.CLOSED:
            return True, "Trading allowed"
        
        elif self.state == CircuitBreakerState.OPEN:
            # Check if cooldown period has passed
            if self.triggered_at:
                time_since_trigger = (pd.Timestamp.now() - self.triggered_at).total_seconds() / 60
                
                if time_since_trigger >= self.cooldown_minutes:
                    # Move to half-open state
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.triggered_at = None
                    logger.info("Circuit breaker moved to HALF_OPEN state")
                    return True, "Limited trading allowed (half-open)"
                else:
                    remaining = self.cooldown_minutes - time_since_trigger
                    return False, f"Circuit breaker open: {remaining:.1f} minutes remaining"
            else:
                return False, "Circuit breaker open"
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Check if we can close the breaker
            if self._can_close_breaker():
                self.state = CircuitBreakerState.CLOSED
                logger.info("Circuit breaker closed - normal trading resumed")
                return True, "Trading allowed"
            else:
                return True, "Limited trading allowed (half-open)"
        
        return False, "Unknown circuit breaker state"
    
    def _trigger_breaker(self, reason: str) -> None:
        """
        Trigger the circuit breaker.
        
        Args:
            reason: Reason for triggering
        """
        if self.state == CircuitBreakerState.CLOSED:
            self.state = CircuitBreakerState.OPEN
            self.triggered_at = pd.Timestamp.now()
            self.trigger_reason = reason
            
            # Add to history
            self.breaker_history.append({
                'timestamp': self.triggered_at,
                'reason': reason,
                'balance': self.current_balance,
                'drawdown': (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0
            })
            
            logger.warning(f"Circuit breaker triggered: {reason}")
    
    def _can_close_breaker(self) -> bool:
        """
        Check if circuit breaker can be closed.
        
        Returns:
            True if breaker can be closed
        """
        # Check if we've recovered from drawdown
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            
            if current_drawdown < self.recovery_threshold:
                return True
        
        # Check if consecutive losses have been reset
        if self.consecutive_losses == 0:
            return True
        
        # Check if daily P&L has recovered
        if self.daily_pnl > 0:
            return True
        
        return False
    
    def reset_daily_pnl(self) -> None:
        """Reset daily P&L (call at start of new day)."""
        self.daily_pnl = 0.0
        logger.info("Daily P&L reset")
    
    def force_close_breaker(self, reason: str = "Manual override") -> None:
        """
        Force close the circuit breaker.
        
        Args:
            reason: Reason for force closing
        """
        self.state = CircuitBreakerState.CLOSED
        self.triggered_at = None
        self.trigger_reason = None
        
        logger.info(f"Circuit breaker force closed: {reason}")
    
    def get_status(self) -> Dict[str, any]:
        """
        Get current circuit breaker status.
        
        Returns:
            Status dictionary
        """
        current_drawdown = 0.0
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        return {
            'state': self.state.value,
            'triggered_at': self.triggered_at,
            'trigger_reason': self.trigger_reason,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown': current_drawdown,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'normal_volatility': self.normal_volatility,
            'total_trades': len(self.trade_history),
            'breaker_triggers': len(self.breaker_history)
        }
    
    def get_breaker_history(self) -> List[Dict[str, any]]:
        """Get circuit breaker trigger history."""
        return self.breaker_history.copy()
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, any]]:
        """Get recent trade history."""
        return self.trade_history[-limit:] if self.trade_history else []
    
    def reset_statistics(self) -> None:
        """Reset all statistics (use with caution)."""
        self.peak_balance = self.current_balance
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trade_history = []
        self.volatility_history = []
        self.breaker_history = []
        
        logger.info("Circuit breaker statistics reset")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CircuitBreaker(state={self.state.value}, drawdown={self.current_drawdown:.2%})"
    
    @property
    def current_drawdown(self) -> float:
        """Get current drawdown percentage."""
        if self.peak_balance > 0:
            return (self.peak_balance - self.current_balance) / self.peak_balance
        return 0.0
