#!/usr/bin/env python3
"""
Backup Script - Pragma Trading Bot

This script creates backups of the Pragma Trading Bot data and configuration.
Supports local and cloud storage options.

Usage:
    python scripts/backup.py --type full
    python scripts/backup.py --type config --destination s3
    python scripts/backup.py --restore --backup-file backup_20241012.tar.gz
"""

import argparse
import sys
import os
import json
import tarfile
import gzip
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BackupManager:
    """Manages backup operations for Pragma Trading Bot"""
    
    def __init__(self, config_path: str = "configs/config-private.json"):
        """
        Initialize backup manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.backup_config = self.config.get('backup', {})
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def create_full_backup(self) -> str:
        """Create full backup of all data"""
        logger.info("Creating full backup...")
        
        backup_name = f"pragma_full_backup_{self.timestamp}"
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        try:
            with tarfile.open(backup_path, "w:gz") as tar:
                # Backup configuration files
                self._add_to_tar(tar, "configs", "configs")
                
                # Backup user data
                if os.path.exists("user_data"):
                    self._add_to_tar(tar, "user_data", "user_data")
                
                # Backup logs
                if os.path.exists("logs"):
                    self._add_to_tar(tar, "logs", "logs")
                
                # Backup source code
                self._add_to_tar(tar, "src", "src")
                
                # Backup scripts
                self._add_to_tar(tar, "scripts", "scripts")
                
                # Backup documentation
                if os.path.exists("docs"):
                    self._add_to_tar(tar, "docs", "docs")
                
                # Backup requirements
                if os.path.exists("requirements.txt"):
                    tar.add("requirements.txt", "requirements.txt")
                
                # Backup pyproject.toml
                if os.path.exists("pyproject.toml"):
                    tar.add("pyproject.toml", "pyproject.toml")
            
            logger.info(f"Full backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error creating full backup: {e}")
            raise
    
    def create_config_backup(self) -> str:
        """Create configuration-only backup"""
        logger.info("Creating configuration backup...")
        
        backup_name = f"pragma_config_backup_{self.timestamp}"
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        try:
            with tarfile.open(backup_path, "w:gz") as tar:
                # Backup configuration files
                self._add_to_tar(tar, "configs", "configs")
                
                # Backup environment files
                if os.path.exists(".env"):
                    tar.add(".env", ".env")
                
                # Backup pyproject.toml
                if os.path.exists("pyproject.toml"):
                    tar.add("pyproject.toml", "pyproject.toml")
            
            logger.info(f"Configuration backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error creating config backup: {e}")
            raise
    
    def create_data_backup(self) -> str:
        """Create data-only backup"""
        logger.info("Creating data backup...")
        
        backup_name = f"pragma_data_backup_{self.timestamp}"
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        try:
            with tarfile.open(backup_path, "w:gz") as tar:
                # Backup user data
                if os.path.exists("user_data"):
                    self._add_to_tar(tar, "user_data", "user_data")
                
                # Backup logs
                if os.path.exists("logs"):
                    self._add_to_tar(tar, "logs", "logs")
            
            logger.info(f"Data backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error creating data backup: {e}")
            raise
    
    def _add_to_tar(self, tar: tarfile.TarFile, source_path: str, arcname: str):
        """Add directory or file to tar archive"""
        if os.path.exists(source_path):
            tar.add(source_path, arcname)
        else:
            logger.warning(f"Path not found: {source_path}")
    
    def upload_to_s3(self, backup_path: str) -> bool:
        """Upload backup to S3"""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            s3_bucket = self.backup_config.get('s3_bucket')
            s3_region = self.backup_config.get('s3_region', 'us-east-1')
            
            if not s3_bucket:
                logger.error("S3 bucket not configured")
                return False
            
            # Initialize S3 client
            s3_client = boto3.client('s3', region_name=s3_region)
            
            # Upload file
            backup_file = Path(backup_path)
            s3_key = f"backups/{backup_file.name}"
            
            logger.info(f"Uploading {backup_file.name} to S3...")
            s3_client.upload_file(backup_path, s3_bucket, s3_key)
            
            logger.info(f"Backup uploaded to S3: s3://{s3_bucket}/{s3_key}")
            return True
            
        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            return False
        except ClientError as e:
            logger.error(f"S3 upload error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        
        try:
            for backup_file in self.backup_dir.glob("*.tar.gz"):
                stat = backup_file.stat()
                backups.append({
                    'name': backup_file.name,
                    'path': str(backup_file),
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x['created'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
        
        return backups
    
    def restore_backup(self, backup_path: str, restore_to: str = ".") -> bool:
        """Restore from backup"""
        logger.info(f"Restoring backup: {backup_path}")
        
        try:
            # Check if backup file exists
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Create restore directory
            restore_dir = Path(restore_to)
            restore_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract backup
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(restore_dir)
            
            logger.info(f"Backup restored to: {restore_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False
    
    def cleanup_old_backups(self, keep_days: int = 30) -> int:
        """Clean up old backups"""
        logger.info(f"Cleaning up backups older than {keep_days} days...")
        
        cleaned_count = 0
        cutoff_time = time.time() - (keep_days * 24 * 3600)
        
        try:
            for backup_file in self.backup_dir.glob("*.tar.gz"):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    cleaned_count += 1
                    logger.info(f"Deleted old backup: {backup_file.name}")
            
            logger.info(f"Cleaned up {cleaned_count} old backups")
            
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
        
        return cleaned_count
    
    def verify_backup(self, backup_path: str) -> bool:
        """Verify backup integrity"""
        logger.info(f"Verifying backup: {backup_path}")
        
        try:
            with tarfile.open(backup_path, "r:gz") as tar:
                # Test if tar file can be opened and read
                members = tar.getmembers()
                logger.info(f"Backup contains {len(members)} files/directories")
                
                # Check for essential files
                essential_files = [
                    "configs/config.json",
                    "user_data",
                    "src"
                ]
                
                missing_files = []
                for essential in essential_files:
                    found = any(member.name.startswith(essential) for member in members)
                    if not found:
                        missing_files.append(essential)
                
                if missing_files:
                    logger.warning(f"Missing essential files: {missing_files}")
                    return False
                
                logger.info("Backup verification passed")
                return True
                
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Backup and restore Pragma Trading Bot"
    )
    
    parser.add_argument(
        "--type",
        choices=["full", "config", "data"],
        default="full",
        help="Type of backup to create"
    )
    
    parser.add_argument(
        "--destination",
        choices=["local", "s3"],
        default="local",
        help="Backup destination"
    )
    
    parser.add_argument(
        "--config",
        default="configs/config-private.json",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available backups"
    )
    
    parser.add_argument(
        "--restore",
        help="Restore from backup file"
    )
    
    parser.add_argument(
        "--verify",
        help="Verify backup file"
    )
    
    parser.add_argument(
        "--cleanup",
        type=int,
        metavar="DAYS",
        help="Clean up backups older than specified days"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize backup manager
        manager = BackupManager(args.config)
        
        if args.list:
            # List backups
            backups = manager.list_backups()
            
            if not backups:
                print("No backups found")
                return 0
            
            print(f"\nFound {len(backups)} backups:")
            print("-" * 80)
            print(f"{'Name':<40} {'Size':<12} {'Created':<20}")
            print("-" * 80)
            
            for backup in backups:
                size_mb = backup['size'] / (1024 * 1024)
                print(f"{backup['name']:<40} {size_mb:>8.1f}MB {backup['created']:<20}")
            
            return 0
        
        elif args.restore:
            # Restore backup
            success = manager.restore_backup(args.restore)
            if success:
                print("✅ Backup restored successfully")
                return 0
            else:
                print("❌ Backup restore failed")
                return 1
        
        elif args.verify:
            # Verify backup
            success = manager.verify_backup(args.verify)
            if success:
                print("✅ Backup verification passed")
                return 0
            else:
                print("❌ Backup verification failed")
                return 1
        
        elif args.cleanup:
            # Cleanup old backups
            cleaned = manager.cleanup_old_backups(args.cleanup)
            print(f"✅ Cleaned up {cleaned} old backups")
            return 0
        
        else:
            # Create backup
            if args.type == "full":
                backup_path = manager.create_full_backup()
            elif args.type == "config":
                backup_path = manager.create_config_backup()
            elif args.type == "data":
                backup_path = manager.create_data_backup()
            else:
                print("❌ Invalid backup type")
                return 1
            
            # Upload to destination
            if args.destination == "s3":
                success = manager.upload_to_s3(backup_path)
                if not success:
                    print("❌ S3 upload failed")
                    return 1
            
            print(f"✅ Backup created: {backup_path}")
            return 0
            
    except Exception as e:
        logger.error(f"Backup error: {e}")
        print(f"❌ Backup error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
