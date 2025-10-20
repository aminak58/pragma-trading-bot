#!/usr/bin/env python3
"""
Paper Trading Validation System
===============================

This module implements a comprehensive paper trading validation system including:
- Paper trading environment setup
- Real-time strategy testing
- Performance validation
- Risk management testing
- System stability validation

Author: Pragma Trading Bot Team
Date: 2025-10-20
"""

import json
import time
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import deque
import requests
import ccxt

@dataclass
class PaperTrade:
    """Paper trade data class"""
    id: str
    symbol: str
    side: str  # buy, sell
    amount: float
    price: float
    timestamp: str
    status: str  # open, closed
    pnl: float = 0.0
    fees: float = 0.0

@dataclass
class PaperPosition:
    """Paper position data class"""
    symbol: str
    side: str
    amount: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: str
    stop_loss: float = 0.0
    take_profit: float = 0.0

@dataclass
class PaperAccount:
    """Paper account data class"""
    balance: float
    positions: List[PaperPosition]
    trades: List[PaperTrade]
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    equity: float
    margin_used: float
    free_margin: float

class PaperTradingEngine:
    """Paper trading engine for strategy validation"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.account = PaperAccount(
            balance=initial_balance,
            positions=[],
            trades=[],
            total_pnl=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            equity=initial_balance,
            margin_used=0.0,
            free_margin=initial_balance
        )
        self.logger = logging.getLogger(__name__)
        self.trade_counter = 0
        self.fee_rate = 0.001  # 0.1% fee
        
    def place_order(self, symbol: str, side: str, amount: float, 
                   price: float, order_type: str = "market") -> str:
        """Place a paper trade order"""
        try:
            self.trade_counter += 1
            trade_id = f"paper_{self.trade_counter}_{int(time.time())}"
            
            # Calculate fees
            fees = amount * price * self.fee_rate
            
            # Check if we have enough balance
            required_balance = amount * price + fees
            if side == "buy" and self.account.free_margin < required_balance:
                self.logger.warning(f"Insufficient balance for buy order: {required_balance}")
                return None
            
            # Create trade
            trade = PaperTrade(
                id=trade_id,
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                timestamp=datetime.now().isoformat(),
                status="open",
                fees=fees
            )
            
            # Update account
            if side == "buy":
                self.account.balance -= required_balance
                self.account.margin_used += required_balance
            else:
                self.account.balance += (amount * price - fees)
                self.account.margin_used -= (amount * price - fees)
            
            # Update positions
            self._update_positions(trade)
            
            # Add to trades
            self.account.trades.append(trade)
            
            # Update account metrics
            self._update_account_metrics()
            
            self.logger.info(f"Paper trade placed: {trade_id} - {side} {amount} {symbol} @ {price}")
            return trade_id
            
        except Exception as e:
            self.logger.error(f"Error placing paper trade: {e}")
            return None
    
    def close_position(self, symbol: str, side: str, amount: float, 
                      price: float) -> bool:
        """Close a paper position"""
        try:
            # Find matching position
            position = None
            for pos in self.account.positions:
                if pos.symbol == symbol and pos.side == side:
                    position = pos
                    break
            
            if not position:
                self.logger.warning(f"No position found to close: {symbol} {side}")
                return False
            
            # Calculate PnL
            if side == "sell":
                pnl = (price - position.entry_price) * amount
            else:
                pnl = (position.entry_price - price) * amount
            
            # Calculate fees
            fees = amount * price * self.fee_rate
            
            # Update account
            self.account.balance += (amount * price - fees)
            self.account.margin_used -= (amount * price - fees)
            self.account.realized_pnl += pnl
            
            # Remove or update position
            if position.amount <= amount:
                self.account.positions.remove(position)
            else:
                position.amount -= amount
            
            # Create closing trade
            trade_id = self.place_order(symbol, "sell" if side == "buy" else "buy", 
                                      amount, price)
            
            self.logger.info(f"Position closed: {symbol} {side} - PnL: {pnl:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False
    
    def update_prices(self, price_data: Dict[str, float]):
        """Update current prices for all positions"""
        try:
            for position in self.account.positions:
                if position.symbol in price_data:
                    position.current_price = price_data[position.symbol]
                    
                    # Calculate unrealized PnL
                    if position.side == "buy":
                        position.unrealized_pnl = (position.current_price - position.entry_price) * position.amount
                    else:
                        position.unrealized_pnl = (position.entry_price - position.current_price) * position.amount
            
            # Update account metrics
            self._update_account_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating prices: {e}")
    
    def _update_positions(self, trade: PaperTrade):
        """Update positions based on trade"""
        try:
            # Find existing position
            existing_position = None
            for pos in self.account.positions:
                if pos.symbol == trade.symbol and pos.side == trade.side:
                    existing_position = pos
                    break
            
            if existing_position:
                # Update existing position
                total_amount = existing_position.amount + trade.amount
                weighted_price = ((existing_position.entry_price * existing_position.amount) + 
                                (trade.price * trade.amount)) / total_amount
                existing_position.amount = total_amount
                existing_position.entry_price = weighted_price
            else:
                # Create new position
                position = PaperPosition(
                    symbol=trade.symbol,
                    side=trade.side,
                    amount=trade.amount,
                    entry_price=trade.price,
                    current_price=trade.price,
                    unrealized_pnl=0.0,
                    entry_time=trade.timestamp
                )
                self.account.positions.append(position)
                
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _update_account_metrics(self):
        """Update account metrics"""
        try:
            # Calculate unrealized PnL
            self.account.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.account.positions)
            
            # Calculate total PnL
            self.account.total_pnl = self.account.realized_pnl + self.account.unrealized_pnl
            
            # Calculate equity
            self.account.equity = self.account.balance + self.account.unrealized_pnl
            
            # Calculate free margin
            self.account.free_margin = self.account.equity - self.account.margin_used
            
        except Exception as e:
            self.logger.error(f"Error updating account metrics: {e}")
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary"""
        return {
            "balance": self.account.balance,
            "equity": self.account.equity,
            "total_pnl": self.account.total_pnl,
            "realized_pnl": self.account.realized_pnl,
            "unrealized_pnl": self.account.unrealized_pnl,
            "margin_used": self.account.margin_used,
            "free_margin": self.account.free_margin,
            "positions_count": len(self.account.positions),
            "trades_count": len(self.account.trades),
            "return_percentage": (self.account.total_pnl / self.initial_balance) * 100
        }

class MarketDataProvider:
    """Market data provider for paper trading"""
    
    def __init__(self, exchange: str = "binance"):
        self.exchange = exchange
        self.logger = logging.getLogger(__name__)
        
        # Initialize exchange
        if exchange == "binance":
            self.exchange_client = ccxt.binance({
                'apiKey': '',
                'secret': '',
                'sandbox': True,  # Use sandbox for paper trading
                'enableRateLimit': True,
            })
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            ticker = self.exchange_client.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols"""
        prices = {}
        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price:
                prices[symbol] = price
        return prices
    
    def get_historical_data(self, symbol: str, timeframe: str = "5m", 
                          limit: int = 100) -> Optional[pd.DataFrame]:
        """Get historical data for symbol"""
        try:
            ohlcv = self.exchange_client.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

class PaperTradingValidator:
    """Paper trading validation system"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.paper_engine = PaperTradingEngine(initial_balance)
        self.market_data = MarketDataProvider()
        self.logger = logging.getLogger(__name__)
        self.validation_results = []
        self.is_running = False
        self.validation_thread = None
        
        # Validation metrics
        self.start_time = None
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        
    def start_validation(self, symbols: List[str], duration_hours: int = 24):
        """Start paper trading validation"""
        if self.is_running:
            self.logger.warning("Validation already running")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        self.validation_thread = threading.Thread(
            target=self._validation_loop,
            args=(symbols, duration_hours),
            daemon=True
        )
        self.validation_thread.start()
        
        self.logger.info(f"Paper trading validation started for {duration_hours} hours")
    
    def stop_validation(self):
        """Stop paper trading validation"""
        self.is_running = False
        if self.validation_thread:
            self.validation_thread.join(timeout=10)
        
        self.logger.info("Paper trading validation stopped")
    
    def _validation_loop(self, symbols: List[str], duration_hours: int):
        """Main validation loop"""
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        while self.is_running and datetime.now() < end_time:
            try:
                # Get current prices
                prices = self.market_data.get_multiple_prices(symbols)
                
                # Update paper trading engine
                self.paper_engine.update_prices(prices)
                
                # Simulate trading decisions (simplified)
                self._simulate_trading_decisions(symbols, prices)
                
                # Update validation metrics
                self._update_validation_metrics()
                
                # Log progress
                if len(self.validation_results) % 10 == 0:
                    self._log_progress()
                
                # Wait before next iteration
                time.sleep(60)  # 1 minute intervals
                
            except Exception as e:
                self.logger.error(f"Error in validation loop: {e}")
                time.sleep(60)
    
    def _simulate_trading_decisions(self, symbols: List[str], prices: Dict[str, float]):
        """Simulate trading decisions (simplified strategy)"""
        try:
            for symbol in symbols:
                if symbol not in prices:
                    continue
                
                current_price = prices[symbol]
                
                # Simple strategy: Buy on price drops, sell on price increases
                # This is just for demonstration - real strategy would be more complex
                
                # Check if we should buy
                if len(self.paper_engine.account.positions) < 3:  # Max 3 positions
                    # Simple buy signal: price below moving average (simplified)
                    if current_price < 50000:  # Simplified condition
                        amount = 0.001  # Small amount for testing
                        self.paper_engine.place_order(symbol, "buy", amount, current_price)
                
                # Check if we should sell
                for position in self.paper_engine.account.positions:
                    if position.symbol == symbol and position.side == "buy":
                        # Simple sell signal: profit target or stop loss
                        profit_pct = (current_price - position.entry_price) / position.entry_price
                        if profit_pct > 0.02 or profit_pct < -0.01:  # 2% profit or 1% loss
                            self.paper_engine.close_position(symbol, "buy", position.amount, current_price)
                
        except Exception as e:
            self.logger.error(f"Error simulating trading decisions: {e}")
    
    def _update_validation_metrics(self):
        """Update validation metrics"""
        try:
            account_summary = self.paper_engine.get_account_summary()
            
            # Update peak balance
            if account_summary['equity'] > self.peak_balance:
                self.peak_balance = account_summary['equity']
            
            # Calculate drawdown
            current_drawdown = (self.peak_balance - account_summary['equity']) / self.peak_balance
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Store validation result
            result = {
                "timestamp": datetime.now().isoformat(),
                "account_summary": account_summary,
                "max_drawdown": self.max_drawdown,
                "validation_duration": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            }
            
            self.validation_results.append(result)
            
        except Exception as e:
            self.logger.error(f"Error updating validation metrics: {e}")
    
    def _log_progress(self):
        """Log validation progress"""
        try:
            if not self.validation_results:
                return
            
            latest_result = self.validation_results[-1]
            account_summary = latest_result['account_summary']
            
            self.logger.info(f"Validation Progress:")
            self.logger.info(f"  Duration: {latest_result['validation_duration']:.0f} seconds")
            self.logger.info(f"  Balance: ${account_summary['balance']:.2f}")
            self.logger.info(f"  Equity: ${account_summary['equity']:.2f}")
            self.logger.info(f"  Total PnL: ${account_summary['total_pnl']:.2f}")
            self.logger.info(f"  Return: {account_summary['return_percentage']:.2f}%")
            self.logger.info(f"  Max Drawdown: {self.max_drawdown:.2%}")
            self.logger.info(f"  Positions: {account_summary['positions_count']}")
            self.logger.info(f"  Trades: {account_summary['trades_count']}")
            
        except Exception as e:
            self.logger.error(f"Error logging progress: {e}")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report"""
        try:
            if not self.validation_results:
                return {"error": "No validation results available"}
            
            latest_result = self.validation_results[-1]
            account_summary = latest_result['account_summary']
            
            # Calculate additional metrics
            total_return = account_summary['return_percentage']
            total_trades = account_summary['trades_count']
            win_rate = self._calculate_win_rate()
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            report = {
                "validation_summary": {
                    "start_time": self.start_time.isoformat() if self.start_time else None,
                    "end_time": datetime.now().isoformat(),
                    "duration_hours": latest_result['validation_duration'] / 3600,
                    "initial_balance": self.initial_balance,
                    "final_balance": account_summary['balance'],
                    "final_equity": account_summary['equity']
                },
                "performance_metrics": {
                    "total_return": total_return,
                    "total_pnl": account_summary['total_pnl'],
                    "realized_pnl": account_summary['realized_pnl'],
                    "unrealized_pnl": account_summary['unrealized_pnl'],
                    "max_drawdown": self.max_drawdown,
                    "sharpe_ratio": sharpe_ratio,
                    "win_rate": win_rate
                },
                "trading_metrics": {
                    "total_trades": total_trades,
                    "positions_count": account_summary['positions_count'],
                    "margin_used": account_summary['margin_used'],
                    "free_margin": account_summary['free_margin']
                },
                "validation_status": {
                    "is_running": self.is_running,
                    "results_count": len(self.validation_results),
                    "data_points": len(self.validation_results)
                },
                "recommendations": self._generate_recommendations(account_summary)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating validation report: {e}")
            return {"error": str(e)}
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trades"""
        try:
            trades = self.paper_engine.account.trades
            if not trades:
                return 0.0
            
            winning_trades = sum(1 for trade in trades if trade.pnl > 0)
            return (winning_trades / len(trades)) * 100
            
        except Exception as e:
            self.logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from returns"""
        try:
            if len(self.validation_results) < 2:
                return 0.0
            
            returns = []
            for i in range(1, len(self.validation_results)):
                prev_equity = self.validation_results[i-1]['account_summary']['equity']
                curr_equity = self.validation_results[i]['account_summary']['equity']
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(ret)
            
            if not returns:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Annualized Sharpe ratio (assuming 1-minute intervals)
            sharpe = (mean_return / std_return) * np.sqrt(525600)  # Minutes in a year
            return sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _generate_recommendations(self, account_summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        try:
            # Return analysis
            return_pct = account_summary['return_percentage']
            if return_pct > 10:
                recommendations.append("High returns achieved - consider reducing position sizes")
            elif return_pct < -5:
                recommendations.append("Negative returns - strategy needs improvement")
            
            # Drawdown analysis
            if self.max_drawdown > 0.15:
                recommendations.append("High drawdown detected - improve risk management")
            elif self.max_drawdown < 0.05:
                recommendations.append("Low drawdown - consider increasing position sizes")
            
            # Trade frequency analysis
            trades_count = account_summary['trades_count']
            if trades_count > 100:
                recommendations.append("High trade frequency - consider reducing signal sensitivity")
            elif trades_count < 10:
                recommendations.append("Low trade frequency - consider relaxing entry conditions")
            
            # Margin usage analysis
            margin_ratio = account_summary['margin_used'] / account_summary['equity']
            if margin_ratio > 0.8:
                recommendations.append("High margin usage - reduce position sizes")
            
            if not recommendations:
                recommendations.append("Validation results look good - ready for live trading")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Error generating recommendations")
        
        return recommendations

def main():
    """Test paper trading validation system"""
    logging.basicConfig(level=logging.INFO)
    
    print("Paper Trading Validation System Test")
    print("=" * 50)
    
    # Create validator
    validator = PaperTradingValidator(initial_balance=10000.0)
    
    # Test symbols
    symbols = ["BTC/USDT", "ETH/USDT"]
    
    print(f"\n1. Starting paper trading validation...")
    print(f"   Initial Balance: ${validator.initial_balance:,.2f}")
    print(f"   Test Symbols: {symbols}")
    print(f"   Duration: 2 minutes (for testing)")
    
    # Start validation
    validator.start_validation(symbols, duration_hours=0.033)  # 2 minutes
    
    # Let it run
    print("   Running validation for 2 minutes...")
    time.sleep(120)  # 2 minutes
    
    # Stop validation
    print("\n2. Stopping validation...")
    validator.stop_validation()
    
    # Get report
    print("\n3. Generating validation report...")
    report = validator.get_validation_report()
    
    if "error" in report:
        print(f"   Error: {report['error']}")
    else:
        print(f"   Validation completed successfully!")
        
        # Summary
        summary = report['validation_summary']
        print(f"\n4. Validation Summary:")
        print(f"   Duration: {summary['duration_hours']:.2f} hours")
        print(f"   Initial Balance: ${summary['initial_balance']:,.2f}")
        print(f"   Final Balance: ${summary['final_balance']:,.2f}")
        print(f"   Final Equity: ${summary['final_equity']:,.2f}")
        
        # Performance
        performance = report['performance_metrics']
        print(f"\n5. Performance Metrics:")
        print(f"   Total Return: {performance['total_return']:.2f}%")
        print(f"   Total PnL: ${performance['total_pnl']:,.2f}")
        print(f"   Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"   Win Rate: {performance['win_rate']:.1f}%")
        
        # Trading
        trading = report['trading_metrics']
        print(f"\n6. Trading Metrics:")
        print(f"   Total Trades: {trading['total_trades']}")
        print(f"   Positions: {trading['positions_count']}")
        print(f"   Margin Used: ${trading['margin_used']:,.2f}")
        print(f"   Free Margin: ${trading['free_margin']:,.2f}")
        
        # Recommendations
        recommendations = report['recommendations']
        print(f"\n7. Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Status
        status = report['validation_status']
        print(f"\n8. Validation Status:")
        print(f"   Running: {status['is_running']}")
        print(f"   Results: {status['results_count']}")
        print(f"   Data Points: {status['data_points']}")
    
    print("\nPaper Trading Validation System test completed!")
    print("\nPaper trading features:")
    print("  - Real-time market data integration")
    print("  - Paper trade execution")
    print("  - Position management")
    print("  - Performance tracking")
    print("  - Risk management validation")
    print("  - Comprehensive reporting")

if __name__ == "__main__":
    main()
