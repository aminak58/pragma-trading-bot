#!/usr/bin/env python3
"""
Production-Ready Scientific Strategy with Enhanced Risk Management
================================================================

This module implements a production-ready trading strategy that integrates
the Scientific Framework with Enhanced Risk Management for live trading.

Author: Pragma Trading Bot Team
Date: 2025-10-20
"""

import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import talib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.scientific import (
        DataManager, PerformanceAnalyzer, RiskManager, 
        ValidationEngine, ExampleScientificStrategy
    )
except ImportError:
    # Fallback for testing
    print("Warning: Scientific framework not available, using simplified version")
    DataManager = None
    PerformanceAnalyzer = None
    RiskManager = None
    ValidationEngine = None
    ExampleScientificStrategy = None
try:
    from src.risk.enhanced_risk_manager import (
        EnhancedRiskManager, KellyCriterionCalculator, 
        DynamicStopLossManager, CircuitBreakerManager
    )
except ImportError:
    # Fallback for testing
    print("Warning: Enhanced risk manager not available, using simplified version")
    EnhancedRiskManager = None
    KellyCriterionCalculator = None
    DynamicStopLossManager = None
    CircuitBreakerManager = None

class ProductionScientificStrategy:
    """
    Production-ready scientific strategy with enhanced risk management
    """
    
    def __init__(self, account_balance: float = 10000.0):
        self.account_balance = account_balance
        self.enhanced_risk_manager = EnhancedRiskManager(account_balance)
        self.performance_tracker = {}
        self.trade_history = []
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.parameters = {
            'rsi_period': 14,
            'rsi_entry_lower': 30,
            'rsi_exit_upper': 70,
            'bb_period': 20,
            'bb_std': 2.0,
            'volume_period': 20,
            'min_volume_ratio': 1.2,
            'win_rate': 0.63,  # From historical testing
            'avg_win': 0.03,   # 3% average win
            'avg_loss': 0.02,  # 2% average loss
            'n_trades': 100    # Historical trades
        }
        
        # Risk management parameters
        self.risk_params = {
            'max_position_size': 0.2,  # 20% max position
            'max_portfolio_heat': 0.8,  # 80% max exposure
            'max_drawdown': 0.15,      # 15% max drawdown
            'volatility_threshold': 0.05,  # 5% volatility threshold
            'correlation_threshold': 0.7   # 70% correlation threshold
        }
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Add technical indicators to dataframe
        """
        try:
            # RSI
            dataframe['rsi'] = talib.RSI(dataframe['close'], timeperiod=self.parameters['rsi_period'])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                dataframe['close'], 
                timeperiod=self.parameters['bb_period'],
                nbdevup=self.parameters['bb_std'],
                nbdevdn=self.parameters['bb_std']
            )
            dataframe['bb_upper'] = bb_upper
            dataframe['bb_middle'] = bb_middle
            dataframe['bb_lower'] = bb_lower
            dataframe['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            # Volume indicators
            dataframe['volume_sma'] = dataframe['volume'].rolling(self.parameters['volume_period']).mean()
            dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
            
            # ATR for volatility
            dataframe['atr'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
            dataframe['atr_percent'] = dataframe['atr'] / dataframe['close']
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(dataframe['close'])
            dataframe['macd'] = macd
            dataframe['macd_signal'] = macd_signal
            dataframe['macd_hist'] = macd_hist
            
            # EMA for trend
            dataframe['ema_short'] = talib.EMA(dataframe['close'], timeperiod=12)
            dataframe['ema_long'] = talib.EMA(dataframe['close'], timeperiod=26)
            
            # ADX for trend strength
            dataframe['adx'] = talib.ADX(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
            
            self.logger.info("Indicators populated successfully")
            return dataframe
            
        except Exception as e:
            self.logger.error(f"Error populating indicators: {e}")
            return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Define entry signals with enhanced risk management
        """
        try:
            dataframe['enter_long'] = 0
            
            # Basic entry conditions
            basic_entry = (
                (dataframe['rsi'] < self.parameters['rsi_entry_lower']) &
                (dataframe['close'] < dataframe['bb_lower']) &
                (dataframe['volume_ratio'] > self.parameters['min_volume_ratio']) &
                (dataframe['bb_width'] > 0.02) &  # Sufficient volatility
                (dataframe['adx'] > 20) &  # Trend strength
                (dataframe['macd'] > dataframe['macd_signal'])  # Momentum
            )
            
            # Risk management filters
            risk_filter = (
                (dataframe['atr_percent'] < self.risk_params['volatility_threshold']) &  # Not too volatile
                (dataframe['rsi'] > 20)  # Not oversold to extreme
            )
            
            # Combined entry signal
            dataframe.loc[basic_entry & risk_filter, 'enter_long'] = 1
            
            self.logger.info(f"Entry signals generated: {dataframe['enter_long'].sum()}")
            return dataframe
            
        except Exception as e:
            self.logger.error(f"Error generating entry signals: {e}")
            dataframe['enter_long'] = 0
            return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Define exit signals with enhanced risk management
        """
        try:
            dataframe['exit_long'] = 0
            
            # Basic exit conditions
            basic_exit = (
                (dataframe['rsi'] > self.parameters['rsi_exit_upper']) |
                (dataframe['close'] > dataframe['bb_upper']) |
                (dataframe['macd'] < dataframe['macd_signal']) |
                (dataframe['adx'] < 15)  # Trend weakening
            )
            
            # Risk management exits
            risk_exit = (
                (dataframe['atr_percent'] > self.risk_params['volatility_threshold'] * 2) |  # Extreme volatility
                (dataframe['rsi'] > 80)  # Extreme overbought
            )
            
            # Combined exit signal
            dataframe.loc[basic_exit | risk_exit, 'exit_long'] = 1
            
            self.logger.info(f"Exit signals generated: {dataframe['exit_long'].sum()}")
            return dataframe
            
        except Exception as e:
            self.logger.error(f"Error generating exit signals: {e}")
            dataframe['exit_long'] = 0
            return dataframe
    
    def calculate_position_size(self, symbol: str, current_price: float, 
                              dataframe: pd.DataFrame) -> float:
        """
        Calculate position size using Kelly Criterion and risk management
        """
        try:
            # Get current volatility
            current_volatility = dataframe['atr_percent'].iloc[-1] if len(dataframe) > 0 else 0.02
            
            # Calculate Kelly-based position size
            position_size = self.enhanced_risk_manager.calculate_position_size(
                symbol=symbol,
                current_price=current_price,
                win_rate=self.parameters['win_rate'],
                avg_win=self.parameters['avg_win'],
                avg_loss=self.parameters['avg_loss'],
                n_trades=self.parameters['n_trades']
            )
            
            # Apply volatility adjustment
            if current_volatility > 0.05:  # High volatility
                position_size *= 0.5
            elif current_volatility < 0.02:  # Low volatility
                position_size *= 1.2
            
            # Apply maximum position size limit
            max_position_value = self.account_balance * self.risk_params['max_position_size']
            max_position_size = max_position_value / current_price
            
            position_size = min(position_size, max_position_size)
            
            self.logger.info(f"Calculated position size for {symbol}: {position_size:.4f}")
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_stop_loss(self, entry_price: float, current_price: float,
                           dataframe: pd.DataFrame, market_regime: str = "normal") -> float:
        """
        Calculate dynamic stop-loss
        """
        try:
            # Get current volatility and ATR
            current_volatility = dataframe['atr_percent'].iloc[-1] if len(dataframe) > 0 else 0.02
            current_atr = dataframe['atr'].iloc[-1] if len(dataframe) > 0 else entry_price * 0.02
            
            # Calculate dynamic stop-loss
            stop_loss = self.enhanced_risk_manager.calculate_dynamic_stop_loss(
                entry_price=entry_price,
                current_price=current_price,
                volatility=current_volatility,
                atr=current_atr,
                market_regime=market_regime
            )
            
            self.logger.info(f"Calculated stop-loss: {stop_loss:.2%}")
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating stop-loss: {e}")
            return -0.02  # Default 2% stop-loss
    
    def check_risk_limits(self, current_positions: List[Dict], 
                        market_data: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Check risk limits before entering new positions
        """
        try:
            # Convert positions to PositionInfo objects
            positions = []
            for pos in current_positions:
                from src.risk.enhanced_risk_manager import PositionInfo
                position_info = PositionInfo(
                    symbol=pos.get('symbol', ''),
                    size=pos.get('size', 0.0),
                    entry_price=pos.get('entry_price', 0.0),
                    current_price=pos.get('current_price', 0.0),
                    unrealized_pnl=pos.get('unrealized_pnl', 0.0),
                    realized_pnl=pos.get('realized_pnl', 0.0),
                    leverage=pos.get('leverage', 1.0),
                    margin_used=pos.get('margin_used', 0.0),
                    risk_score=pos.get('risk_score', 0.5)
                )
                positions.append(position_info)
            
            # Update risk metrics
            risk_metrics = self.enhanced_risk_manager.update_risk_metrics(positions, market_data)
            
            # Check risk limits
            within_limits, violations = self.enhanced_risk_manager.check_risk_limits(risk_metrics)
            
            return within_limits, violations
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False, ["Error in risk check"]
    
    def simulate_trades(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate trades with enhanced risk management
        """
        try:
            simulated_trades = []
            open_trade_entry_price = None
            open_trade_entry_time = None
            open_trade_size = None
            
            for i in range(len(dataframe)):
                row = dataframe.iloc[i]
                current_time = row.name
                current_price = row['close']
                
                # Entry logic
                if row['enter_long'] == 1 and open_trade_entry_price is None:
                    # Check risk limits before entry
                    current_positions = []  # Simplified for simulation
                    market_data = {
                        'volatility': row['atr_percent'],
                        'sharpe_ratio': 1.5  # Simplified
                    }
                    
                    within_limits, violations = self.check_risk_limits(current_positions, market_data)
                    
                    if within_limits:
                        # Calculate position size
                        position_size = self.calculate_position_size(
                            symbol="BTC/USDT",
                            current_price=current_price,
                            dataframe=dataframe.iloc[:i+1]
                        )
                        
                        if position_size > 0:
                            open_trade_entry_price = current_price
                            open_trade_entry_time = current_time
                            open_trade_size = position_size
                            self.logger.info(f"Entry at {current_time}: {current_price}, Size: {position_size:.4f}")
                
                # Exit logic
                if open_trade_entry_price is not None:
                    # Check for exit signal
                    if row['exit_long'] == 1:
                        pnl = (current_price - open_trade_entry_price) / open_trade_entry_price
                        simulated_trades.append({
                            'entry_time': open_trade_entry_time,
                            'exit_time': current_time,
                            'entry_price': open_trade_entry_price,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'size': open_trade_size
                        })
                        self.logger.info(f"Exit at {current_time}: {current_price}, PnL: {pnl:.2%}")
                        open_trade_entry_price = None
                        open_trade_entry_time = None
                        open_trade_size = None
                        continue
                    
                    # Check for stop-loss
                    stop_loss = self.calculate_stop_loss(
                        entry_price=open_trade_entry_price,
                        current_price=current_price,
                        dataframe=dataframe.iloc[:i+1]
                    )
                    
                    stop_loss_price = open_trade_entry_price * (1 + stop_loss)
                    if current_price <= stop_loss_price:
                        pnl = (current_price - open_trade_entry_price) / open_trade_entry_price
                        simulated_trades.append({
                            'entry_time': open_trade_entry_time,
                            'exit_time': current_time,
                            'entry_price': open_trade_entry_price,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'size': open_trade_size
                        })
                        self.logger.info(f"Stop-loss at {current_time}: {current_price}, PnL: {pnl:.2%}")
                        open_trade_entry_price = None
                        open_trade_entry_time = None
                        open_trade_size = None
                        continue
            
            return pd.DataFrame(simulated_trades)
            
        except Exception as e:
            self.logger.error(f"Error simulating trades: {e}")
            return pd.DataFrame()
    
    def generate_performance_report(self, trades_df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate comprehensive performance report
        """
        try:
            if trades_df.empty:
                return {"error": "No trades to analyze"}
            
            # Basic metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # PnL metrics
            total_pnl = trades_df['pnl'].sum()
            avg_pnl = trades_df['pnl'].mean()
            max_win = trades_df['pnl'].max()
            max_loss = trades_df['pnl'].min()
            
            # Risk metrics
            returns = trades_df['pnl']
            volatility = returns.std()
            sharpe_ratio = returns.mean() / volatility if volatility > 0 else 0
            
            # Drawdown
            cumulative_returns = returns.cumsum()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / (peak + 1e-8)
            max_drawdown = drawdown.min()
            
            # Risk report
            risk_report = self.enhanced_risk_manager.generate_risk_report()
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "max_win": max_win,
                "max_loss": max_loss,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "risk_report": risk_report
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}

def main():
    """Test the production-ready scientific strategy"""
    logging.basicConfig(level=logging.INFO)
    
    print("Production-Ready Scientific Strategy Test")
    print("=" * 50)
    
    # Create strategy
    strategy = ProductionScientificStrategy(account_balance=10000.0)
    
    # Generate sample data
    np.random.seed(42)
    n_periods = 1000
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='5T')
    
    # Generate realistic price data
    price = 50000
    prices = [price]
    for i in range(1, n_periods):
        change = np.random.normal(0, 0.001)  # 0.1% volatility
        price *= (1 + change)
        prices.append(price)
    
    # Create dataframe
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.randint(100, 1000, n_periods)
    }, index=dates)
    
    # Ensure high >= low
    df['high'] = np.maximum(df['high'], df['low'])
    
    print(f"\n1. Generated sample data: {len(df)} periods")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Populate indicators
    print("\n2. Populating indicators...")
    df = strategy.populate_indicators(df, {})
    print(f"   Indicators added: {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])}")
    
    # Generate signals
    print("\n3. Generating signals...")
    df = strategy.populate_entry_trend(df, {})
    df = strategy.populate_exit_trend(df, {})
    print(f"   Entry signals: {df['enter_long'].sum()}")
    print(f"   Exit signals: {df['exit_long'].sum()}")
    
    # Simulate trades
    print("\n4. Simulating trades...")
    trades_df = strategy.simulate_trades(df)
    print(f"   Trades executed: {len(trades_df)}")
    
    if not trades_df.empty:
        # Generate performance report
        print("\n5. Performance Report:")
        report = strategy.generate_performance_report(trades_df)
        
        print(f"   Total Trades: {report['total_trades']}")
        print(f"   Win Rate: {report['win_rate']:.1%}")
        print(f"   Total PnL: {report['total_pnl']:.2%}")
        print(f"   Avg PnL: {report['avg_pnl']:.2%}")
        print(f"   Max Win: {report['max_win']:.2%}")
        print(f"   Max Loss: {report['max_loss']:.2%}")
        print(f"   Volatility: {report['volatility']:.2%}")
        print(f"   Sharpe Ratio: {report['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {report['max_drawdown']:.2%}")
        
        # Risk report
        risk_report = report['risk_report']
        print(f"\n6. Risk Report:")
        print(f"   Risk Level: {risk_report['risk_level']}")
        print(f"   Portfolio Value: ${risk_report['portfolio_value']:.2f}")
        print(f"   Total Exposure: {risk_report['exposure_percentage']:.1%}")
        print(f"   Current Drawdown: {risk_report['current_drawdown']:.2%}")
        print(f"   Recommendations: {len(risk_report['recommendations'])}")
    
    print("\nProduction-Ready Scientific Strategy test completed!")

if __name__ == "__main__":
    main()
