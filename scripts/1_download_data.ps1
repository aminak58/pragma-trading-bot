# Pragma Trading Bot - Data Download Script
# Downloads historical market data for backtesting

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pragma Trading Bot - Data Download" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Freqtrade is installed
$freqtradeInstalled = Get-Command freqtrade -ErrorAction SilentlyContinue
if (-not $freqtradeInstalled) {
    Write-Host "❌ ERROR: Freqtrade not found!" -ForegroundColor Red
    Write-Host "Please install Freqtrade first:" -ForegroundColor Yellow
    Write-Host "  pip install freqtrade" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Freqtrade found" -ForegroundColor Green
Write-Host ""

# Configuration
$EXCHANGE = "binance"
$TIMEFRAMES = "5m"
$DAYS = 180
$PAIRS = @(
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "MATIC/USDT",
    "LINK/USDT",
    "XRP/USDT"
)

Write-Host "📊 Download Configuration:" -ForegroundColor Cyan
Write-Host "  Exchange: $EXCHANGE"
Write-Host "  Timeframe: $TIMEFRAMES"
Write-Host "  Days: $DAYS"
Write-Host "  Pairs: $($PAIRS.Count)"
Write-Host ""

foreach ($pair in $PAIRS) {
    Write-Host "  - $pair"
}
Write-Host ""

# Confirm
$confirm = Read-Host "Download data? (y/n)"
if ($confirm -ne "y") {
    Write-Host "❌ Cancelled" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "⬇️  Downloading data..." -ForegroundColor Cyan
Write-Host ""

# Download data
$pairsString = $PAIRS -join " "

try {
    & freqtrade download-data `
        --exchange $EXCHANGE `
        --pairs $pairsString `
        --timeframes $TIMEFRAMES `
        --days $DAYS `
        --data-format-ohlcv json `
        --datadir user_data/data
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Data download complete!" -ForegroundColor Green
        Write-Host ""
        Write-Host "📁 Data location: user_data/data/" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Next step: Run backtest" -ForegroundColor Yellow
        Write-Host "  .\scripts\2_run_backtest.ps1" -ForegroundColor Yellow
    } else {
        Write-Host ""
        Write-Host "❌ Download failed!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host ""
    Write-Host "❌ Error: $_" -ForegroundColor Red
    exit 1
}
