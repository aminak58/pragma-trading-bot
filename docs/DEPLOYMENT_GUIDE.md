# Deployment Guide: Pragma Trading Bot

**نسخه:** 1.0  
**تاریخ:** 2025-10-12  
**وضعیت:** 📋 Complete

---

## 🚀 Quick Start Deployment

### Prerequisites

- Python 3.11+
- Docker (optional)
- Freqtrade 2025.9.1+
- Binance API credentials

### Option 1: Docker Deployment (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/your-username/pragma-trading-bot.git
cd pragma-trading-bot

# 2. Copy configuration
cp configs/pragma_config.example.json configs/config-private.json

# 3. Edit configuration with your API keys
nano configs/config-private.json

# 4. Build and run
docker-compose build
docker-compose up -d pragma-bot

# 5. Check logs
docker-compose logs -f pragma-bot
```

### Option 2: Local Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/pragma-trading-bot.git
cd pragma-trading-bot

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy strategy to Freqtrade
cp src/strategies/regime_adaptive_strategy.py /path/to/freqtrade/user_data/strategies/
cp -r src/regime /path/to/freqtrade/user_data/strategies/
cp -r src/risk /path/to/freqtrade/user_data/strategies/
cp -r src/ml /path/to/freqtrade/user_data/strategies/

# 5. Download data
cd /path/to/freqtrade
freqtrade download-data --exchange binance --pairs BTC/USDT ETH/USDT --timeframes 5m --days 180

# 6. Run backtest
freqtrade backtesting --strategy RegimeAdaptiveStrategy --config /path/to/pragma-trading-bot/configs/config-private.json --timerange 20240701-20241010

# 7. Run live trading (dry run)
freqtrade trade --strategy RegimeAdaptiveStrategy --config /path/to/pragma-trading-bot/configs/config-private.json
```

---

## ⚙️ Configuration

### 1. API Configuration

```json
{
  "exchange": {
    "name": "binance",
    "key": "YOUR_BINANCE_API_KEY",
    "secret": "YOUR_BINANCE_SECRET_KEY",
    "ccxt_config": {},
    "ccxt_async_config": {}
  }
}
```

### 2. Risk Management Settings

```json
{
  "max_open_trades": 5,
  "stake_currency": "USDT",
  "stake_amount": "unlimited",
  "tradable_balance_ratio": 0.99,
  "dry_run": true,
  "dry_run_wallet": 1000
}
```

### 3. Strategy Parameters

```json
{
  "strategy": "RegimeAdaptiveStrategy",
  "timeframe": "5m",
  "process_only_new_candles": true,
  "use_exit_signal": true,
  "exit_profit_only": false
}
```

---

## 🔧 Production Setup

### 1. System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB SSD
- OS: Ubuntu 20.04+ / Windows 10+ / macOS 10.15+

**Recommended:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+ SSD
- Network: Stable internet connection

### 2. Security Configuration

```bash
# Create dedicated user
sudo useradd -m -s /bin/bash freqtrade
sudo usermod -aG docker freqtrade

# Set up SSH keys
sudo -u freqtrade ssh-keygen -t rsa -b 4096

# Configure firewall
sudo ufw allow 22/tcp
sudo ufw allow 8080/tcp  # For API server
sudo ufw enable
```

### 3. Environment Variables

```bash
# Create .env file
cat > .env << EOF
FREQTRADE_CONFIG=/path/to/config-private.json
FREQTRADE_STRATEGY=RegimeAdaptiveStrategy
FREQTRADE_STRATEGY_PATH=/path/to/strategies
FREQTRADE_DATA=/path/to/data
FREQTRADE_USERDATA=/path/to/user_data
FREQTRADE_LOGS=/path/to/logs
EOF
```

### 4. Systemd Service (Linux)

```bash
# Create service file
sudo tee /etc/systemd/system/pragma-bot.service > /dev/null << EOF
[Unit]
Description=Pragma Trading Bot
After=network.target

[Service]
Type=simple
User=freqtrade
Group=freqtrade
WorkingDirectory=/path/to/pragma-trading-bot
Environment=PATH=/path/to/pragma-trading-bot/venv/bin
ExecStart=/path/to/pragma-trading-bot/venv/bin/freqtrade trade --config /path/to/config-private.json
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable pragma-bot
sudo systemctl start pragma-bot
```

---

## 📊 Monitoring & Logging

### 1. Log Configuration

```json
{
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "/path/to/logs/pragma-bot.log",
    "max_bytes": 10485760,
    "backup_count": 5
  }
}
```

### 2. Performance Monitoring

```bash
# Monitor system resources
htop
iotop
nethogs

# Monitor bot logs
tail -f /path/to/logs/pragma-bot.log

# Monitor trades
freqtrade show-trades --config /path/to/config-private.json
```

### 3. Health Checks

```bash
# Check bot status
freqtrade status --config /path/to/config-private.json

# Check open trades
freqtrade show-trades --config /path/to/config-private.json

# Check performance
freqtrade profit --config /path/to/config-private.json
```

---

## 🔄 Maintenance

### 1. Regular Tasks

**Daily:**
- Check bot status
- Review trade performance
- Monitor system resources

**Weekly:**
- Review logs for errors
- Check model performance
- Update data if needed

**Monthly:**
- Retrain ML models
- Review and update parameters
- Backup configuration and data

### 2. Model Retraining

```bash
# Automatic retraining (every 15 days)
python scripts/retrain_models.py --config /path/to/config-private.json

# Manual retraining
python scripts/retrain_models.py --force --config /path/to/config-private.json
```

### 3. Backup Strategy

```bash
# Create backup script
cat > backup.sh << EOF
#!/bin/bash
DATE=\$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/path/to/backups/\$DATE"

mkdir -p \$BACKUP_DIR
cp -r /path/to/pragma-trading-bot \$BACKUP_DIR/
cp /path/to/config-private.json \$BACKUP_DIR/
cp -r /path/to/freqtrade/user_data \$BACKUP_DIR/

# Keep only last 30 backups
find /path/to/backups -type d -mtime +30 -exec rm -rf {} \;
EOF

chmod +x backup.sh

# Schedule daily backup
crontab -e
# Add: 0 2 * * * /path/to/backup.sh
```

---

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Check Python path
export PYTHONPATH="/path/to/pragma-trading-bot/src:$PYTHONPATH"

# Verify installation
python -c "from regime.hmm_detector import RegimeDetector; print('OK')"
```

**2. Memory Issues**
```bash
# Increase swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**3. API Connection Issues**
```bash
# Test API connection
freqtrade test-pairlist --config /path/to/config-private.json

# Check API limits
freqtrade show-config --config /path/to/config-private.json
```

**4. Model Training Issues**
```bash
# Check data availability
ls -la /path/to/freqtrade/user_data/data/

# Verify feature engineering
python -c "from src.ml.feature_engineering import FeatureEngineer; print('OK')"
```

### Debug Mode

```bash
# Run with debug logging
freqtrade trade --strategy RegimeAdaptiveStrategy --config /path/to/config-private.json --logfile /path/to/debug.log --verbosity 3
```

---

## 📈 Performance Optimization

### 1. System Optimization

```bash
# CPU optimization
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Memory optimization
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

### 2. Freqtrade Optimization

```json
{
  "internals": {
    "process_throttle_secs": 5,
    "heartbeat_interval": 60
  },
  "api_server": {
    "enabled": true,
    "listen_ip_address": "127.0.0.1",
    "listen_port": 8080
  }
}
```

### 3. Database Optimization

```bash
# Use PostgreSQL for better performance
pip install psycopg2-binary

# Configure database
freqtrade create-userdir --userdir /path/to/user_data
```

---

## 🔐 Security Best Practices

### 1. API Security

- Use API keys with limited permissions
- Enable IP whitelisting on exchange
- Rotate API keys regularly
- Never commit API keys to version control

### 2. System Security

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install security tools
sudo apt install fail2ban ufw

# Configure fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 3. Data Security

```bash
# Encrypt sensitive data
gpg --symmetric --cipher-algo AES256 config-private.json

# Use encrypted storage
sudo cryptsetup luksFormat /dev/sdb1
sudo cryptsetup luksOpen /dev/sdb1 encrypted_storage
```

---

## 📞 Support

### Getting Help

1. **Documentation:** Check this guide and API reference
2. **Issues:** Create GitHub issue with logs
3. **Community:** Freqtrade Discord/Telegram
4. **Professional:** Contact for paid support

### Log Collection

```bash
# Collect logs for support
tar -czf pragma-bot-logs-$(date +%Y%m%d).tar.gz \
  /path/to/logs/ \
  /path/to/freqtrade/user_data/logs/ \
  /var/log/syslog
```

---

**Last Updated:** 2025-10-12  
**Next Review:** After major updates
