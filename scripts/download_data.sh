#!/bin/bash
# Quick data download script for validation

echo "======================================"
echo "Downloading Historical Market Data"
echo "======================================"
echo ""

# Configuration
EXCHANGE="binance"
TIMEFRAME="5m"
DAYS=180
PAIRS="BTC/USDT ETH/USDT BNB/USDT SOL/USDT ADA/USDT"

echo "Exchange: $EXCHANGE"
echo "Timeframe: $TIMEFRAME"
echo "Days: $DAYS"
echo "Pairs: $PAIRS"
echo ""
echo "Starting download..."
echo ""

freqtrade download-data \
  --exchange $EXCHANGE \
  --pairs $PAIRS \
  --timeframes $TIMEFRAME \
  --days $DAYS \
  --data-format-ohlcv json \
  --datadir user_data/data

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Download complete!"
    echo ""
    echo "Data saved to: user_data/data/"
    echo ""
else
    echo ""
    echo "❌ Download failed!"
    exit 1
fi
