# Pragma Trading Bot - Backtest Execution Script
# Runs comprehensive backtest with RegimeAdaptiveStrategy

param(
    [string]$Timerange = "20240701-20241010",
    [string]$Config = "configs/backtest_config.example.json",
    [switch]$Detailed = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pragma Trading Bot - Backtest" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Freqtrade is installed
$freqtradeInstalled = Get-Command freqtrade -ErrorAction SilentlyContinue
if (-not $freqtradeInstalled) {
    Write-Host "❌ ERROR: Freqtrade not found!" -ForegroundColor Red
    exit 1
}

# Check if strategy file exists
$strategyPath = "src/strategies/regime_adaptive_strategy.py"
if (-not (Test-Path $strategyPath)) {
    Write-Host "❌ ERROR: Strategy file not found: $strategyPath" -ForegroundColor Red
    exit 1
}

# Check if config exists
if (-not (Test-Path $Config)) {
    Write-Host "❌ ERROR: Config file not found: $Config" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Strategy: RegimeAdaptiveStrategy" -ForegroundColor Green
Write-Host "✅ Config: $Config" -ForegroundColor Green
Write-Host "✅ Timerange: $Timerange" -ForegroundColor Green
Write-Host ""

# Copy strategy to user_data (if needed)
$userDataStrategy = "user_data/strategies/regime_adaptive_strategy.py"
if (-not (Test-Path $userDataStrategy)) {
    Write-Host "📋 Copying strategy to user_data..." -ForegroundColor Cyan
    Copy-Item $strategyPath $userDataStrategy -Force
    
    # Copy regime module
    $regimeSource = "src/regime"
    $regimeDest = "user_data/strategies/regime"
    if (Test-Path $regimeSource) {
        Copy-Item $regimeSource $regimeDest -Recurse -Force
        Write-Host "✅ Strategy and dependencies copied" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "🚀 Starting backtest..." -ForegroundColor Cyan
Write-Host ""

# Build command
$backtestCmd = "freqtrade backtesting " +
    "--strategy RegimeAdaptiveStrategy " +
    "--config $Config " +
    "--timerange $Timerange " +
    "--export trades"

if ($Detailed) {
    $backtestCmd += " --breakdown day week month"
}

# Run backtest
try {
    Write-Host "Command: $backtestCmd" -ForegroundColor Gray
    Write-Host ""
    
    Invoke-Expression $backtestCmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "✅ Backtest Complete!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "📊 Results saved to: user_data/backtest_results/" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "  1. Review results above" -ForegroundColor Yellow
        Write-Host "  2. Run analysis: .\scripts\3_analyze_results.ps1" -ForegroundColor Yellow
        Write-Host "  3. If good, run hyperopt: .\scripts\4_run_hyperopt.ps1" -ForegroundColor Yellow
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "❌ Backtest failed!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host ""
    Write-Host "❌ Error: $_" -ForegroundColor Red
    exit 1
}
