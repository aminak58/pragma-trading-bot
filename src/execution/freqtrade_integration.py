"""
Freqtrade Integration Module

Provides integration between Pragma execution layer and Freqtrade.
Handles command building, process management, and result parsing.
"""

import subprocess
import logging
import os
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class FreqtradeIntegration:
    """
    Integration class for Freqtrade execution.
    
    Handles:
    - Command building for different modes
    - Process management
    - Result parsing
    - Error handling
    """
    
    def __init__(self, config: Dict[str, Any], strategy_path: str = "src/strategies"):
        """
        Initialize Freqtrade integration.
        
        Args:
            config: Strategy configuration
            strategy_path: Path to strategy files
        """
        self.config = config
        self.strategy_path = strategy_path
        self.process = None
        self.is_running = False
        
        # Validate Freqtrade installation
        self._validate_freqtrade_installation()
    
    def _validate_freqtrade_installation(self):
        """Validate that Freqtrade is installed and accessible."""
        try:
            result = subprocess.run(
                ["freqtrade", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"Freqtrade version: {result.stdout.strip()}")
            else:
                raise RuntimeError(f"Freqtrade not working: {result.stderr}")
                
        except FileNotFoundError:
            raise RuntimeError(
                "Freqtrade not found. Please install Freqtrade first:\n"
                "pip install freqtrade"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Freqtrade command timed out")
        except Exception as e:
            raise RuntimeError(f"Error validating Freqtrade: {e}")
    
    def build_command(self, mode: str, config_path: str, **kwargs) -> List[str]:
        """
        Build Freqtrade command for specified mode.
        
        Args:
            mode: Execution mode (backtest, dry-run, live)
            config_path: Path to configuration file
            **kwargs: Additional command arguments
            
        Returns:
            List of command arguments
        """
        cmd = ["freqtrade", "trade"]
        
        # Add strategy
        cmd.extend(["--strategy", "RegimeAdaptiveStrategy"])
        
        # Add config
        cmd.extend(["--config", config_path])
        
        # Add strategy path
        if self.strategy_path:
            cmd.extend(["--strategy-path", self.strategy_path])
        
        # Mode-specific arguments
        if mode == "backtest":
            cmd = ["freqtrade", "backtesting"]
            cmd.extend(["--strategy", "RegimeAdaptiveStrategy"])
            cmd.extend(["--config", config_path])
            
            # Add timerange
            timerange = kwargs.get('timerange', '20240101-20241001')
            cmd.extend(["--timerange", timerange])
            
            # Add export options
            cmd.extend(["--export", "trades"])
            cmd.extend(["--export-filename", "backtest_results"])
            
        elif mode == "dry-run":
            cmd.append("--dry-run")
            
        elif mode == "live":
            # Live trading - no additional flags needed
            pass
        
        # Add common arguments
        if self.config.get('timeframe'):
            cmd.extend(["--timeframe", self.config['timeframe']])
        
        if self.config.get('pairs'):
            cmd.extend(["--pairs"] + self.config['pairs'])
        
        # Add custom arguments
        for key, value in kwargs.items():
            if key not in ['timerange']:  # Already handled
                if isinstance(value, bool) and value:
                    cmd.append(f"--{key}")
                elif not isinstance(value, bool):
                    cmd.extend([f"--{key}", str(value)])
        
        return cmd
    
    def start_execution(self, mode: str, config_path: str, **kwargs) -> bool:
        """
        Start Freqtrade execution.
        
        Args:
            mode: Execution mode
            config_path: Path to configuration file
            **kwargs: Additional arguments
            
        Returns:
            True if started successfully
        """
        try:
            # Build command
            cmd = self.build_command(mode, config_path, **kwargs)
            
            logger.info(f"Starting Freqtrade execution: {' '.join(cmd)}")
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.is_running = True
            
            # Log start
            logger.info(f"Freqtrade process started with PID: {self.process.pid}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting Freqtrade execution: {e}")
            return False
    
    def monitor_execution(self, timeout: Optional[int] = None) -> Tuple[bool, str]:
        """
        Monitor execution and return results.
        
        Args:
            timeout: Maximum execution time in seconds
            
        Returns:
            Tuple of (success, output)
        """
        if not self.process:
            return False, "No process running"
        
        start_time = time.time()
        output_lines = []
        
        try:
            while True:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning("Execution timeout reached")
                    self.stop_execution()
                    return False, "Execution timeout"
                
                # Read output
                if self.process.stdout.readable():
                    line = self.process.stdout.readline()
                    if line:
                        output_lines.append(line.strip())
                        print(line.strip())  # Real-time output
                
                # Check if process finished
                if self.process.poll() is not None:
                    break
                
                # Small delay to prevent busy waiting
                time.sleep(0.1)
            
            # Get final output
            remaining_output = self.process.stdout.read()
            if remaining_output:
                output_lines.append(remaining_output.strip())
            
            # Get stderr
            stderr_output = self.process.stderr.read()
            if stderr_output:
                logger.warning(f"Freqtrade stderr: {stderr_output}")
                output_lines.append(f"STDERR: {stderr_output}")
            
            # Check return code
            return_code = self.process.returncode
            success = return_code == 0
            
            if success:
                logger.info("Freqtrade execution completed successfully")
            else:
                logger.error(f"Freqtrade execution failed with return code: {return_code}")
            
            return success, "\n".join(output_lines)
            
        except Exception as e:
            logger.error(f"Error monitoring execution: {e}")
            return False, str(e)
    
    def stop_execution(self) -> bool:
        """
        Stop Freqtrade execution.
        
        Returns:
            True if stopped successfully
        """
        if not self.process:
            return True
        
        try:
            logger.info("Stopping Freqtrade execution...")
            
            # Send SIGTERM
            self.process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                logger.warning("Force killing Freqtrade process")
                self.process.kill()
                self.process.wait()
            
            self.is_running = False
            logger.info("Freqtrade execution stopped")
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping execution: {e}")
            return False
    
    def get_execution_status(self) -> Dict[str, Any]:
        """
        Get current execution status.
        
        Returns:
            Dictionary with status information
        """
        status = {
            'is_running': self.is_running,
            'process_id': self.process.pid if self.process else None,
            'return_code': self.process.returncode if self.process else None
        }
        
        return status
    
    def parse_backtest_results(self, results_path: str) -> Dict[str, Any]:
        """
        Parse backtest results from file.
        
        Args:
            results_path: Path to results file
            
        Returns:
            Dictionary with parsed results
        """
        try:
            if not os.path.exists(results_path):
                logger.warning(f"Results file not found: {results_path}")
                return {}
            
            # Parse JSON results
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Extract key metrics
            parsed_results = {
                'total_trades': len(results.get('trades', [])),
                'total_profit': sum(trade.get('profit_abs', 0) for trade in results.get('trades', [])),
                'win_rate': self._calculate_win_rate(results.get('trades', [])),
                'max_drawdown': self._calculate_max_drawdown(results.get('trades', [])),
                'sharpe_ratio': self._calculate_sharpe_ratio(results.get('trades', [])),
                'trades': results.get('trades', [])
            }
            
            return parsed_results
            
        except Exception as e:
            logger.error(f"Error parsing backtest results: {e}")
            return {}
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0
        
        winning_trades = [t for t in trades if t.get('profit_abs', 0) > 0]
        return len(winning_trades) / len(trades)
    
    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown from trades."""
        if not trades:
            return 0.0
        
        cumulative_profit = 0
        max_profit = 0
        max_drawdown = 0
        
        for trade in trades:
            cumulative_profit += trade.get('profit_abs', 0)
            max_profit = max(max_profit, cumulative_profit)
            drawdown = max_profit - cumulative_profit
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """Calculate Sharpe ratio from trades."""
        if not trades:
            return 0.0
        
        profits = [t.get('profit_abs', 0) for t in trades]
        
        if not profits:
            return 0.0
        
        import numpy as np
        
        mean_profit = np.mean(profits)
        std_profit = np.std(profits)
        
        if std_profit == 0:
            return 0.0
        
        # Assuming risk-free rate of 0 for simplicity
        return mean_profit / std_profit
    
    def cleanup(self):
        """Cleanup resources."""
        if self.process:
            self.stop_execution()
        
        self.is_running = False
        self.process = None


class FreqtradeManager:
    """
    High-level manager for Freqtrade operations.
    
    Provides convenient methods for common operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Freqtrade manager.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.integration = FreqtradeIntegration(config)
    
    def run_backtest(self, 
                    config_path: str, 
                    timerange: str = "20240101-20241001",
                    **kwargs) -> Dict[str, Any]:
        """
        Run backtest.
        
        Args:
            config_path: Path to configuration file
            timerange: Time range for backtest
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest for timerange: {timerange}")
        
        # Start execution
        success = self.integration.start_execution(
            mode="backtest",
            config_path=config_path,
            timerange=timerange,
            **kwargs
        )
        
        if not success:
            return {"success": False, "error": "Failed to start backtest"}
        
        # Monitor execution
        success, output = self.integration.monitor_execution(timeout=3600)  # 1 hour timeout
        
        if not success:
            return {"success": False, "error": output}
        
        # Parse results
        results_path = "backtest_results.json"
        parsed_results = self.integration.parse_backtest_results(results_path)
        
        return {
            "success": True,
            "output": output,
            "results": parsed_results
        }
    
    def run_dry_run(self, config_path: str, **kwargs) -> Dict[str, Any]:
        """
        Run dry-run trading.
        
        Args:
            config_path: Path to configuration file
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with dry-run results
        """
        logger.info("Starting dry-run trading")
        
        # Start execution
        success = self.integration.start_execution(
            mode="dry-run",
            config_path=config_path,
            **kwargs
        )
        
        if not success:
            return {"success": False, "error": "Failed to start dry-run"}
        
        # Monitor execution
        success, output = self.integration.monitor_execution()
        
        return {
            "success": success,
            "output": output
        }
    
    def run_live_trading(self, config_path: str, **kwargs) -> Dict[str, Any]:
        """
        Run live trading.
        
        Args:
            config_path: Path to configuration file
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with live trading results
        """
        logger.critical("Starting live trading")
        
        # Start execution
        success = self.integration.start_execution(
            mode="live",
            config_path=config_path,
            **kwargs
        )
        
        if not success:
            return {"success": False, "error": "Failed to start live trading"}
        
        # Monitor execution
        success, output = self.integration.monitor_execution()
        
        return {
            "success": success,
            "output": output
        }
    
    def stop_trading(self) -> bool:
        """
        Stop trading.
        
        Returns:
            True if stopped successfully
        """
        return self.integration.stop_execution()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get trading status.
        
        Returns:
            Dictionary with status information
        """
        return self.integration.get_execution_status()
    
    def cleanup(self):
        """Cleanup resources."""
        self.integration.cleanup()
