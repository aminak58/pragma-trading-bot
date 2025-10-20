"""
Scientific Trading Strategy Framework - Risk Manager

This module implements the RiskManager class for comprehensive risk management,
monitoring, and alerting in scientific strategy validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

class RiskManager:
    """
    Risk management and monitoring for scientific strategy validation
    
    This class handles:
    - Position sizing and portfolio heat management
    - Correlation risk monitoring
    - Volatility-adjusted risk controls
    - Real-time risk alerts and monitoring
    """
    
    def __init__(self, max_position_size: float = 0.05, 
                 max_portfolio_heat: float = 0.20,
                 max_correlation: float = 0.7,
                 volatility_lookback: int = 20):
        """
        Initialize RiskManager
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_portfolio_heat: Maximum portfolio heat (total exposure)
            max_correlation: Maximum correlation between positions
            volatility_lookback: Lookback period for volatility calculation
        """
        self.max_position_size = max_position_size
        self.max_portfolio_heat = max_portfolio_heat
        self.max_correlation = max_correlation
        self.volatility_lookback = volatility_lookback
        
        self.risk_metrics = {}
        self.alerts = []
        self.logger = logging.getLogger(__name__)
        
        # Risk monitoring history
        self.risk_history = []
    
    def calculate_risk_metrics(self, positions: pd.DataFrame, 
                             returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics
        
        Args:
            positions: DataFrame containing position information
            returns: Series of returns
            
        Returns:
            Dictionary of risk metrics
        """
        self.logger.info("Calculating risk metrics...")
        
        # Position sizing risk
        position_metrics = self.calculate_position_metrics(positions)
        
        # Portfolio heat risk
        portfolio_metrics = self.calculate_portfolio_metrics(positions)
        
        # Correlation risk
        correlation_metrics = self.calculate_correlation_metrics(positions)
        
        # Volatility risk
        volatility_metrics = self.calculate_volatility_metrics(returns)
        
        # Concentration risk
        concentration_metrics = self.calculate_concentration_metrics(positions)
        
        # Combine all metrics
        self.risk_metrics = {
            'position_metrics': position_metrics,
            'portfolio_metrics': portfolio_metrics,
            'correlation_metrics': correlation_metrics,
            'volatility_metrics': volatility_metrics,
            'concentration_metrics': concentration_metrics,
            'timestamp': datetime.now()
        }
        
        # Check risk limits
        self.check_risk_limits()
        
        # Store in history
        self.risk_history.append(self.risk_metrics.copy())
        
        self.logger.info("Risk metrics calculated successfully")
        return self.risk_metrics
    
    def calculate_position_metrics(self, positions: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate position sizing risk metrics
        
        Args:
            positions: DataFrame containing position information
            
        Returns:
            Position risk metrics
        """
        if len(positions) == 0:
            return {
                'num_positions': 0,
                'max_position_size': 0,
                'avg_position_size': 0,
                'position_size_std': 0,
                'position_size_violations': 0
            }
        
        # Calculate position sizes (assuming 'size' column exists)
        if 'size' in positions.columns:
            position_sizes = positions['size'].abs()
        else:
            # Estimate position sizes from PnL if size not available
            position_sizes = positions['pnl'].abs() / positions['pnl'].abs().sum()
        
        return {
            'num_positions': len(positions),
            'max_position_size': position_sizes.max(),
            'avg_position_size': position_sizes.mean(),
            'position_size_std': position_sizes.std(),
            'position_size_violations': (position_sizes > self.max_position_size).sum(),
            'position_size_distribution': position_sizes.describe().to_dict()
        }
    
    def calculate_portfolio_metrics(self, positions: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate portfolio heat and exposure metrics
        
        Args:
            positions: DataFrame containing position information
            
        Returns:
            Portfolio risk metrics
        """
        if len(positions) == 0:
            return {
                'portfolio_heat': 0,
                'total_exposure': 0,
                'net_exposure': 0,
                'long_exposure': 0,
                'short_exposure': 0
            }
        
        # Calculate exposures
        if 'size' in positions.columns:
            total_exposure = positions['size'].abs().sum()
            net_exposure = positions['size'].sum()
            long_exposure = positions[positions['size'] > 0]['size'].sum()
            short_exposure = abs(positions[positions['size'] < 0]['size'].sum())
        else:
            # Estimate from PnL
            total_exposure = positions['pnl'].abs().sum()
            net_exposure = positions['pnl'].sum()
            long_exposure = positions[positions['pnl'] > 0]['pnl'].sum()
            short_exposure = abs(positions[positions['pnl'] < 0]['pnl'].sum())
        
        # Calculate portfolio heat
        portfolio_heat = total_exposure / max(1, total_exposure)  # Normalize
        
        return {
            'portfolio_heat': portfolio_heat,
            'total_exposure': total_exposure,
            'net_exposure': net_exposure,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'exposure_ratio': long_exposure / max(1, short_exposure)
        }
    
    def calculate_correlation_metrics(self, positions: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate correlation risk metrics
        
        Args:
            positions: DataFrame containing position information
            
        Returns:
            Correlation risk metrics
        """
        if len(positions) < 2:
            return {
                'max_correlation': 0,
                'avg_correlation': 0,
                'correlation_violations': 0,
                'correlation_matrix': {}
            }
        
        # Calculate correlations between positions
        # This is a simplified version - in practice, you'd use actual price data
        if 'pnl' in positions.columns and len(positions) > 1:
            # Use PnL as proxy for returns
            pnl_data = positions['pnl'].values.reshape(-1, 1)
            if pnl_data.shape[0] > 1:
                correlation_matrix = np.corrcoef(pnl_data.T)
                
                # Extract upper triangle (excluding diagonal)
                if hasattr(correlation_matrix, 'shape') and len(correlation_matrix.shape) > 1 and correlation_matrix.shape[0] > 1:
                    upper_triangle = np.triu(correlation_matrix, k=1)
                    correlations = upper_triangle[upper_triangle != 0]
                else:
                    correlations = np.array([])
                
                if len(correlations) > 0:
                    max_correlation = np.max(np.abs(correlations))
                    avg_correlation = np.mean(np.abs(correlations))
                    violations = (np.abs(correlations) > self.max_correlation).sum()
                else:
                    max_correlation = 0
                    avg_correlation = 0
                    violations = 0
            else:
                max_correlation = 0
                avg_correlation = 0
                violations = 0
                correlation_matrix = {}
        else:
            max_correlation = 0
            avg_correlation = 0
            violations = 0
            correlation_matrix = {}
        
        return {
            'max_correlation': max_correlation,
            'avg_correlation': avg_correlation,
            'correlation_violations': violations,
            'correlation_matrix': correlation_matrix.tolist() if hasattr(correlation_matrix, 'tolist') else correlation_matrix
        }
    
    def calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate volatility risk metrics
        
        Args:
            returns: Series of returns
            
        Returns:
            Volatility risk metrics
        """
        if len(returns) == 0:
            return {
                'current_volatility': 0,
                'volatility_percentile': 0,
                'volatility_trend': 0,
                'volatility_regime': 'unknown'
            }
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=self.volatility_lookback).std()
        current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0
        
        # Calculate volatility percentile
        vol_percentile = (rolling_vol < current_vol).sum() / len(rolling_vol) if len(rolling_vol) > 0 else 0.5
        
        # Calculate volatility trend
        if len(rolling_vol) >= 5:
            vol_trend = np.polyfit(range(5), rolling_vol.tail(5), 1)[0]
        else:
            vol_trend = 0
        
        # Determine volatility regime
        if vol_percentile > 0.8:
            vol_regime = 'high'
        elif vol_percentile < 0.2:
            vol_regime = 'low'
        else:
            vol_regime = 'normal'
        
        return {
            'current_volatility': current_vol,
            'volatility_percentile': vol_percentile,
            'volatility_trend': vol_trend,
            'volatility_regime': vol_regime,
            'rolling_volatility': rolling_vol.tail(10).tolist()
        }
    
    def calculate_concentration_metrics(self, positions: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate concentration risk metrics
        
        Args:
            positions: DataFrame containing position information
            
        Returns:
            Concentration risk metrics
        """
        if len(positions) == 0:
            return {
                'herfindahl_index': 0,
                'concentration_ratio': 0,
                'max_concentration': 0
            }
        
        # Calculate position weights
        if 'size' in positions.columns:
            weights = positions['size'].abs()
        else:
            weights = positions['pnl'].abs()
        
        total_weight = weights.sum()
        if total_weight > 0:
            weights = weights / total_weight
        else:
            weights = pd.Series([0] * len(positions))
        
        # Calculate Herfindahl-Hirschman Index
        hhi = (weights ** 2).sum()
        
        # Calculate concentration ratio (top 5 positions)
        top_5_weight = weights.nlargest(5).sum()
        
        # Maximum concentration
        max_concentration = weights.max()
        
        return {
            'herfindahl_index': hhi,
            'concentration_ratio': top_5_weight,
            'max_concentration': max_concentration,
            'effective_positions': 1 / hhi if hhi > 0 else 0
        }
    
    def check_risk_limits(self) -> List[str]:
        """
        Check if risk limits are exceeded
        
        Returns:
            List of risk alert messages
        """
        alerts = []
        
        # Check position size limits
        position_metrics = self.risk_metrics.get('position_metrics', {})
        if position_metrics.get('max_position_size', 0) > self.max_position_size:
            alerts.append(f"Position size exceeded: {position_metrics['max_position_size']:.2%} > {self.max_position_size:.2%}")
        
        # Check portfolio heat limits
        portfolio_metrics = self.risk_metrics.get('portfolio_metrics', {})
        if portfolio_metrics.get('portfolio_heat', 0) > self.max_portfolio_heat:
            alerts.append(f"Portfolio heat exceeded: {portfolio_metrics['portfolio_heat']:.2%} > {self.max_portfolio_heat:.2%}")
        
        # Check correlation limits
        correlation_metrics = self.risk_metrics.get('correlation_metrics', {})
        if correlation_metrics.get('max_correlation', 0) > self.max_correlation:
            alerts.append(f"High correlation detected: {correlation_metrics['max_correlation']:.2f} > {self.max_correlation:.2f}")
        
        # Check concentration limits
        concentration_metrics = self.risk_metrics.get('concentration_metrics', {})
        if concentration_metrics.get('max_concentration', 0) > 0.3:  # 30% max concentration
            alerts.append(f"High concentration: {concentration_metrics['max_concentration']:.2%} > 30%")
        
        # Check volatility limits
        volatility_metrics = self.risk_metrics.get('volatility_metrics', {})
        if volatility_metrics.get('volatility_regime') == 'high':
            alerts.append("High volatility regime detected")
        
        self.alerts = alerts
        
        if alerts:
            self.logger.warning(f"Risk alerts triggered: {len(alerts)} alerts")
            for alert in alerts:
                self.logger.warning(f"  - {alert}")
        
        return alerts
    
    def adjust_position_sizes(self, positions: pd.DataFrame, 
                            target_volatility: float = 0.15) -> pd.DataFrame:
        """
        Adjust position sizes based on volatility
        
        Args:
            positions: DataFrame containing position information
            target_volatility: Target portfolio volatility
            
        Returns:
            DataFrame with adjusted position sizes
        """
        if len(positions) == 0:
            return positions.copy()
        
        adjusted_positions = positions.copy()
        
        # Calculate current volatility
        volatility_metrics = self.risk_metrics.get('volatility_metrics', {})
        current_volatility = volatility_metrics.get('current_volatility', 0.1)
        
        # Calculate volatility adjustment factor
        vol_adjustment = target_volatility / max(current_volatility, 0.01)
        
        # Adjust position sizes
        if 'size' in adjusted_positions.columns:
            adjusted_positions['size'] = adjusted_positions['size'] * vol_adjustment
        else:
            # Create size column from PnL
            adjusted_positions['size'] = adjusted_positions['pnl'] * vol_adjustment
        
        # Ensure position sizes don't exceed limits
        if 'size' in adjusted_positions.columns:
            max_size = adjusted_positions['size'].abs().max()
            if max_size > self.max_position_size:
                scale_factor = self.max_position_size / max_size
                adjusted_positions['size'] = adjusted_positions['size'] * scale_factor
        
        return adjusted_positions
    
    def calculate_portfolio_heat_limit(self, current_volatility: float) -> float:
        """
        Calculate dynamic portfolio heat limit based on volatility
        
        Args:
            current_volatility: Current market volatility
            
        Returns:
            Adjusted portfolio heat limit
        """
        # Reduce heat limit in high volatility environments
        if current_volatility > 0.2:  # High volatility
            return self.max_portfolio_heat * 0.5
        elif current_volatility > 0.15:  # Medium volatility
            return self.max_portfolio_heat * 0.75
        else:  # Low volatility
            return self.max_portfolio_heat
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk summary
        
        Returns:
            Risk summary dictionary
        """
        return {
            'risk_metrics': self.risk_metrics,
            'alerts': self.alerts,
            'risk_score': self.calculate_risk_score(),
            'recommendations': self.get_risk_recommendations(),
            'timestamp': datetime.now()
        }
    
    def calculate_risk_score(self) -> float:
        """
        Calculate overall risk score (0-1, higher is riskier)
        
        Returns:
            Risk score between 0 and 1
        """
        risk_score = 0.0
        
        # Position size risk
        position_metrics = self.risk_metrics.get('position_metrics', {})
        max_position_size = position_metrics.get('max_position_size', 0)
        risk_score += min(max_position_size / self.max_position_size, 1.0) * 0.3
        
        # Portfolio heat risk
        portfolio_metrics = self.risk_metrics.get('portfolio_metrics', {})
        portfolio_heat = portfolio_metrics.get('portfolio_heat', 0)
        risk_score += min(portfolio_heat / self.max_portfolio_heat, 1.0) * 0.3
        
        # Correlation risk
        correlation_metrics = self.risk_metrics.get('correlation_metrics', {})
        max_correlation = correlation_metrics.get('max_correlation', 0)
        risk_score += min(max_correlation / self.max_correlation, 1.0) * 0.2
        
        # Concentration risk
        concentration_metrics = self.risk_metrics.get('concentration_metrics', {})
        max_concentration = concentration_metrics.get('max_concentration', 0)
        risk_score += min(max_concentration / 0.3, 1.0) * 0.2
        
        return min(risk_score, 1.0)
    
    def get_risk_recommendations(self) -> List[str]:
        """
        Get risk management recommendations
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Position size recommendations
        position_metrics = self.risk_metrics.get('position_metrics', {})
        if position_metrics.get('max_position_size', 0) > self.max_position_size:
            recommendations.append("Reduce maximum position size")
        
        # Portfolio heat recommendations
        portfolio_metrics = self.risk_metrics.get('portfolio_metrics', {})
        if portfolio_metrics.get('portfolio_heat', 0) > self.max_portfolio_heat:
            recommendations.append("Reduce overall portfolio exposure")
        
        # Correlation recommendations
        correlation_metrics = self.risk_metrics.get('correlation_metrics', {})
        if correlation_metrics.get('max_correlation', 0) > self.max_correlation:
            recommendations.append("Diversify positions to reduce correlation")
        
        # Concentration recommendations
        concentration_metrics = self.risk_metrics.get('concentration_metrics', {})
        if concentration_metrics.get('max_concentration', 0) > 0.3:
            recommendations.append("Reduce position concentration")
        
        # Volatility recommendations
        volatility_metrics = self.risk_metrics.get('volatility_metrics', {})
        if volatility_metrics.get('volatility_regime') == 'high':
            recommendations.append("Consider reducing exposure in high volatility environment")
        
        return recommendations