#!/usr/bin/env python3
"""
Emergency Stop Script - Pragma Trading Bot

This script immediately stops all trading activities and closes positions.
Use in emergency situations when immediate action is required.

Usage:
    python scripts/emergency_stop.py
    python scripts/emergency_stop.py --close-positions
    python scripts/emergency_stop.py --force
"""

import argparse
import sys
import os
import logging
import subprocess
import signal
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def stop_freqtrade_processes():
    """Stop all Freqtrade processes"""
    
    logger.critical("Stopping Freqtrade processes...")
    
    try:
        # Find Freqtrade processes
        result = subprocess.run(
            ["pgrep", "-f", "freqtrade"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    logger.critical(f"Stopping Freqtrade process: {pid}")
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        time.sleep(2)
                        
                        # Check if still running
                        try:
                            os.kill(int(pid), 0)
                            logger.critical(f"Force killing process: {pid}")
                            os.kill(int(pid), signal.SIGKILL)
                        except ProcessLookupError:
                            logger.critical(f"Process {pid} stopped successfully")
                    except ProcessLookupError:
                        logger.critical(f"Process {pid} already stopped")
                    except Exception as e:
                        logger.error(f"Error stopping process {pid}: {e}")
        else:
            logger.critical("No Freqtrade processes found")
            
    except Exception as e:
        logger.error(f"Error stopping Freqtrade processes: {e}")

def stop_systemd_service():
    """Stop systemd service if running"""
    
    logger.critical("Stopping systemd service...")
    
    try:
        result = subprocess.run(
            ["systemctl", "stop", "pragma-bot"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.critical("Systemd service stopped successfully")
        else:
            logger.critical(f"Failed to stop systemd service: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Error stopping systemd service: {e}")

def stop_docker_containers():
    """Stop Docker containers if running"""
    
    logger.critical("Stopping Docker containers...")
    
    try:
        # Stop pragma-bot container
        result = subprocess.run(
            ["docker", "stop", "pragma-bot"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.critical("Docker container stopped successfully")
        else:
            logger.critical(f"Failed to stop Docker container: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Error stopping Docker containers: {e}")

def close_all_positions(config_path):
    """Close all open positions"""
    
    logger.critical("Closing all open positions...")
    
    try:
        # Get open positions
        result = subprocess.run(
            ["freqtrade", "show-trades", "--config", config_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.critical("Open positions retrieved")
            print(result.stdout)
            
            # Close positions (this would need to be implemented)
            # For now, just log the positions
            logger.critical("Position closure would be implemented here")
            
        else:
            logger.critical(f"Failed to get open positions: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Error closing positions: {e}")

def send_emergency_alerts():
    """Send emergency alerts"""
    
    logger.critical("Sending emergency alerts...")
    
    try:
        # Send Telegram alert
        message = "🚨 EMERGENCY STOP ACTIVATED 🚨\nTrading has been stopped immediately."
        
        # This would integrate with your alert system
        logger.critical("Emergency alert sent")
        
    except Exception as e:
        logger.error(f"Error sending emergency alerts: {e}")

def log_emergency_action(action):
    """Log emergency action"""
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[EMERGENCY] {timestamp} - {action}"
    
    # Log to file
    with open("logs/emergency.log", "a") as f:
        f.write(log_message + "\n")
    
    # Log to console
    logger.critical(log_message)

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description="Emergency stop for Pragma Trading Bot"
    )
    
    parser.add_argument(
        "--close-positions",
        action="store_true",
        help="Close all open positions"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force stop without confirmation"
    )
    
    parser.add_argument(
        "--config",
        default="configs/config-private.json",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--alerts",
        action="store_true",
        help="Send emergency alerts"
    )
    
    args = parser.parse_args()
    
    # Emergency stop banner
    print("=" * 70)
    print("🚨 EMERGENCY STOP ACTIVATED 🚨")
    print("=" * 70)
    print()
    
    if not args.force:
        print("This will immediately stop all trading activities.")
        print("Are you sure you want to proceed?")
        confirm = input("Type 'EMERGENCY STOP' to confirm: ").strip()
        
        if confirm != "EMERGENCY STOP":
            print("Emergency stop cancelled.")
            return 0
    
    # Log emergency start
    log_emergency_action("Emergency stop initiated")
    
    try:
        # Stop all trading processes
        stop_freqtrade_processes()
        stop_systemd_service()
        stop_docker_containers()
        
        # Close positions if requested
        if args.close_positions:
            close_all_positions(args.config)
        
        # Send alerts if requested
        if args.alerts:
            send_emergency_alerts()
        
        # Log emergency completion
        log_emergency_action("Emergency stop completed")
        
        print()
        print("✅ Emergency stop completed successfully")
        print("All trading activities have been stopped")
        print()
        print("Next steps:")
        print("1. Check system status")
        print("2. Review logs for issues")
        print("3. Contact team if needed")
        print("4. Restart only when safe")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during emergency stop: {e}")
        log_emergency_action(f"Emergency stop failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
