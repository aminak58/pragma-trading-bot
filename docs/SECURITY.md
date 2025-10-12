# Security Guidelines

**Critical: Protecting API Keys, Credentials, and Sensitive Trading Data**

---

## Table of Contents

1. [Overview](#overview)
2. [Credential Management](#credential-management)
3. [API Key Security](#api-key-security)
4. [Configuration Best Practices](#configuration-best-practices)
5. [Production Deployment](#production-deployment)
6. [Network Security](#network-security)
7. [Docker Security](#docker-security)
8. [Incident Response](#incident-response)
9. [Security Checklist](#security-checklist)

---

## Overview

This document outlines security best practices for the Pragma Trading Bot. **Never commit secrets, API keys, or private configurations to version control.**

### Security Principles

1. **Secrets in Environment Variables**: Never hardcode credentials
2. **Principle of Least Privilege**: Minimal API permissions
3. **Defense in Depth**: Multiple security layers
4. **Regular Audits**: Continuous security reviews
5. **Secure by Default**: Safe configuration templates

---

## Credential Management

### ❌ **NEVER DO THIS**

```python
# ❌ Hardcoded credentials in code
API_KEY = "pk_live_abc123xyz"
API_SECRET = "sk_live_secret123"

# ❌ Committed config files
# config-private.json with real keys in git
```

### ✅ **DO THIS**

```python
# ✅ Use environment variables
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

API_KEY = os.getenv('EXCHANGE_API_KEY')
API_SECRET = os.getenv('EXCHANGE_API_SECRET')

if not API_KEY or not API_SECRET:
    raise ValueError("API credentials not found in environment")
```

### Environment Variable Setup

#### Create `.env` file (NEVER commit this)

```bash
# .env - Keep this file LOCAL and SECRET

# Exchange API Credentials
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_API_SECRET=your_api_secret_here

# Telegram Bot (optional)
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Database (if using external DB)
DATABASE_URL=postgresql://user:password@localhost:5432/trading_db

# API Server
API_JWT_SECRET=generate_random_secret_here
API_USERNAME=admin
API_PASSWORD=strong_password_here
```

#### Add to `.gitignore`

```bash
# In .gitignore (already added)
.env
.env.local
.env.*.local
config-private.json
*-private.json
*-secret.json
*.key
*.pem
```

---

## API Key Security

### Exchange API Key Best Practices

#### 1. **Use Read/Trade-Only Keys**

**Never grant withdrawal permissions to trading bot API keys.**

Binance/Exchange Settings:
- ✅ Enable Reading
- ✅ Enable Spot & Margin Trading (if needed)
- ❌ **DISABLE Withdrawals**
- ❌ **DISABLE Universal Transfer** (if possible)
- ✅ Restrict by IP address

#### 2. **IP Whitelisting**

```json
{
  "_comment": "Whitelist your server IPs",
  "allowed_ips": [
    "203.0.113.1",
    "203.0.113.2"
  ]
}
```

Configure on exchange:
1. Go to API Management
2. Edit API key
3. Enable "Restrict access to trusted IPs only"
4. Add your VPS/server IP addresses

#### 3. **Separate Keys for Testing**

```
Production Key:  (Live trading, limited permissions)
  └─ api_key_prod
  └─ Withdrawal: DISABLED
  └─ IP whitelist: Enabled

Testnet/Paper Key: (Testing, no real funds)
  └─ api_key_test
  └─ Testnet.binance.com
```

#### 4. **Key Rotation**

- Rotate API keys every 90 days
- Immediately rotate if compromised
- Log all key generation/deletion events

---

## Configuration Best Practices

### Config File Structure

```
configs/
├── pragma_config.example.json     # ✅ Committed (no secrets)
├── backtest_config.example.json   # ✅ Committed (no secrets)
├── config-private.json            # ❌ NEVER commit (real keys)
└── README.md                      # ✅ Instructions
```

### Safe Config Template

```json
{
  "_comment": "config-private.json - KEEP THIS FILE SECRET",
  "_instructions": "Copy from pragma_config.example.json and fill in real credentials",
  
  "exchange": {
    "name": "binance",
    "key": "${EXCHANGE_API_KEY}",    
    "secret": "${EXCHANGE_API_SECRET}"
  },
  
  "telegram": {
    "enabled": true,
    "token": "${TELEGRAM_TOKEN}",
    "chat_id": "${TELEGRAM_CHAT_ID}"
  },
  
  "api_server": {
    "enabled": true,
    "jwt_secret_key": "${API_JWT_SECRET}",
    "username": "${API_USERNAME}",
    "password": "${API_PASSWORD}"
  }
}
```

### Loading Config Securely

```python
import json
import os
from string import Template

def load_secure_config(config_path: str) -> dict:
    """
    Load config and substitute environment variables.
    """
    with open(config_path, 'r') as f:
        config_text = f.read()
    
    # Substitute ${VAR} with environment variables
    template = Template(config_text)
    config_text = template.safe_substitute(os.environ)
    
    config = json.loads(config_text)
    
    # Validate required secrets are present
    validate_config_security(config)
    
    return config

def validate_config_security(config: dict):
    """
    Ensure no placeholder values remain.
    """
    def check_placeholders(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                check_placeholders(v, f"{path}.{k}")
        elif isinstance(obj, str):
            if "${" in obj or "YOUR_" in obj or obj == "":
                raise ValueError(
                    f"Placeholder found at {path}: {obj}. "
                    "Set environment variable."
                )
    
    check_placeholders(config)
```

---

## Production Deployment

### Pre-Deployment Security Checklist

```bash
# 1. Check for secrets in git history
git log --all --full-history --source --  -- '*password*' '*secret*' '*key*'

# 2. Scan for hardcoded credentials
grep -r "api_key\s*=\s*['\"]" src/
grep -r "password\s*=\s*['\"]" src/

# 3. Check .gitignore
cat .gitignore | grep -E '\.env|private|secret|key'

# 4. Verify no secrets in configs
grep -r "YOUR_" configs/
grep -r "secret.*:.*[a-zA-Z0-9]" configs/*.json

# 5. Check file permissions
ls -la configs/config-private.json  # Should be 600 (rw-------)
```

### Secure Deployment Script

```bash
#!/bin/bash
# deploy_secure.sh

set -e  # Exit on error

echo "🔒 Pragma Trading Bot - Secure Deployment"
echo "=========================================="

# 1. Check environment variables
required_vars=(
    "EXCHANGE_API_KEY"
    "EXCHANGE_API_SECRET"
    "TELEGRAM_TOKEN"
    "API_JWT_SECRET"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ ERROR: $var not set"
        exit 1
    fi
    echo "✅ $var is set"
done

# 2. Set secure file permissions
echo ""
echo "🔒 Setting secure file permissions..."
chmod 600 .env
chmod 600 configs/config-private.json
echo "✅ Permissions set"

# 3. Verify no secrets in git
echo ""
echo "🔍 Checking for secrets in repository..."
if git grep -i "api_key.*=.*['\"][a-z0-9]" HEAD; then
    echo "❌ SECRETS FOUND IN GIT! Aborting."
    exit 1
fi
echo "✅ No secrets found in git"

# 4. Run security scan
echo ""
echo "🔍 Running security scan..."
pip install bandit safety
bandit -r src/ -f screen || true
safety check --json || true

# 5. Start bot
echo ""
echo "🚀 Starting bot..."
docker-compose up -d pragma-bot

echo ""
echo "✅ Deployment complete!"
```

---

## Network Security

### Firewall Rules

```bash
# UFW (Ubuntu Firewall) setup

# Default: Deny all incoming, allow outgoing
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (change port if non-standard)
sudo ufw allow 22/tcp

# Allow Freqtrade API (only from specific IPs)
sudo ufw allow from 203.0.113.0/24 to any port 8080

# Enable firewall
sudo ufw enable
```

### HTTPS/TLS for API

```python
# Use HTTPS for API server
config = {
    "api_server": {
        "enabled": true,
        "listen_ip_address": "0.0.0.0",
        "listen_port": 8080,
        "verbosity": "error",
        
        # Enable HTTPS
        "tls": {
            "enabled": true,
            "certificate": "/path/to/cert.pem",
            "private_key": "/path/to/key.pem"
        },
        
        # JWT authentication
        "jwt_secret_key": "${API_JWT_SECRET}",
        "username": "${API_USERNAME}",
        "password": "${API_PASSWORD}"
    }
}
```

---

## Docker Security

### Secure Dockerfile Practices

```dockerfile
# ✅ Run as non-root user
RUN groupadd -r freqtrade && \
    useradd -r -g freqtrade -u 1000 freqtrade
USER freqtrade

# ✅ Minimal base image
FROM python:3.11-slim-bullseye  # Not 'latest'

# ✅ No secrets in build args
# ARG API_KEY  # ❌ NEVER DO THIS
# Use environment variables at runtime instead

# ✅ Read-only root filesystem (where possible)
docker run --read-only pragma-bot

# ✅ Drop capabilities
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE pragma-bot
```

### Docker Secrets (Swarm/Kubernetes)

```yaml
# docker-compose.yml with secrets
version: '3.8'
services:
  pragma-bot:
    image: pragma-trading-bot:latest
    secrets:
      - api_key
      - api_secret
    environment:
      - EXCHANGE_API_KEY_FILE=/run/secrets/api_key
      - EXCHANGE_API_SECRET_FILE=/run/secrets/api_secret

secrets:
  api_key:
    file: ./secrets/api_key.txt
  api_secret:
    file: ./secrets/api_secret.txt
```

---

## Incident Response

### If API Key is Compromised

**Immediate Actions:**

1. **Revoke the key immediately** on exchange
2. **Check account history** for unauthorized trades
3. **Generate new key** with restricted permissions
4. **Rotate all related credentials**
5. **Review logs** for breach source
6. **Update IP whitelist** if needed

### Monitoring for Compromise

```python
def check_for_suspicious_activity(trades: list) -> list:
    """
    Detect potentially unauthorized trades.
    """
    alerts = []
    
    for trade in trades:
        # Check for unusual patterns
        if trade['amount'] > NORMAL_MAX_AMOUNT * 10:
            alerts.append(f"Large trade detected: {trade}")
        
        if trade['pair'] not in EXPECTED_PAIRS:
            alerts.append(f"Unexpected pair: {trade}")
        
        if is_outside_trading_hours(trade['timestamp']):
            alerts.append(f"Trade outside hours: {trade}")
    
    return alerts
```

---

## Security Checklist

### Development

- [ ] No hardcoded credentials in code
- [ ] `.env` added to `.gitignore`
- [ ] Environment variables used for secrets
- [ ] Security linters run (bandit, safety)
- [ ] Dependencies scanned for vulnerabilities
- [ ] Code reviews include security check

### Deployment

- [ ] API keys have withdrawal disabled
- [ ] IP whitelist configured on exchange
- [ ] Separate keys for test/production
- [ ] File permissions set correctly (600 for secrets)
- [ ] Firewall rules configured
- [ ] HTTPS enabled for API
- [ ] Non-root user in Docker
- [ ] Secrets not in git history
- [ ] Monitoring/alerts configured
- [ ] Incident response plan documented

### Operations

- [ ] API keys rotated quarterly
- [ ] Regular security audits
- [ ] Dependency updates applied
- [ ] Logs reviewed for anomalies
- [ ] Backups encrypted
- [ ] 2FA enabled on all accounts
- [ ] Access logs maintained

---

## Tools & Resources

### Security Scanning

```bash
# Install security tools
pip install bandit safety pip-audit

# Scan for vulnerabilities
bandit -r src/ -f json -o security-report.json
safety check --json
pip-audit
```

### Secret Detection

```bash
# Install gitleaks
# https://github.com/gitleaks/gitleaks

# Scan git history for secrets
gitleaks detect --source . --verbose

# Pre-commit hook
gitleaks protect --staged
```

### Encryption

```bash
# Encrypt sensitive files
# Using GPG
gpg -c config-private.json  # Encrypt
gpg config-private.json.gpg  # Decrypt

# Using age (modern alternative)
age -p config-private.json > config-private.json.age
age -d config-private.json.age > config-private.json
```

---

## Emergency Contacts

**Security Issues:**
- Report to: [security@your-domain.com]
- Response time: < 24 hours

**Exchange Support:**
- Binance: https://www.binance.com/en/support
- Emergency: Use chat for urgent issues

---

**Last Updated**: 2025-10-12  
**Version**: 1.0  
**Maintainer**: Pragma Trading Team  
**Review Frequency**: Quarterly
