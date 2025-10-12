# Pragma Trading Bot - Hyperopt Execution Script
# Optimizes strategy parameters using Freqtrade hyperopt

param(
    [int]$Epochs = 100,
    [string]$Loss = "SharpeHyperOptLoss",
    [string]$Spaces = "buy sell",
    [string]$Timerange = "20240701-20241010"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pragma Trading Bot - Hyperopt" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "⚙️  Configuration:" -ForegroundColor Cyan
Write-Host "  Epochs: $Epochs"
Write-Host "  Loss Function: $Loss"
Write-Host "  Spaces: $Spaces"
Write-Host "  Timerange: $Timerange"
Write-Host ""

Write-Host "⚠️  WARNING: Hyperopt can take several hours!" -ForegroundColor Yellow
Write-Host ""

# Available loss functions
Write-Host "📊 Available Loss Functions:" -ForegroundColor Cyan
Write-Host "  - SharpeHyperOptLoss (Recommended - maximize Sharpe ratio)"
Write-Host "  - SortinoHyperOptLoss (Downside risk focus)"
Write-Host "  - MaxDrawDownHyperOptLoss (Minimize drawdown)"
Write-Host "  - CalmarHyperOptLoss (Calmar ratio)"
Write-Host "  - OnlyProfitHyperOptLoss (Simple profit)"
Write-Host ""

# Confirm
$confirm = Read-Host "Continue with hyperopt? (y/n)"
if ($confirm -ne "y") {
    Write-Host "❌ Cancelled" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "🚀 Starting hyperopt..." -ForegroundColor Cyan
Write-Host "   This may take 2-4 hours for 100 epochs" -ForegroundColor Yellow
Write-Host ""

try {
    $hyperoptCmd = "freqtrade hyperopt " +
        "--hyperopt-loss $Loss " +
        "--strategy RegimeAdaptiveStrategy " +
        "--config configs/backtest_config.example.json " +
        "--timerange $Timerange " +
        "--epochs $Epochs " +
        "--spaces $Spaces " +
        "--random-state 42"
    
    Write-Host "Command: $hyperoptCmd" -ForegroundColor Gray
    Write-Host ""
    
    Invoke-Expression $hyperoptCmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "✅ Hyperopt Complete!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "📊 Results saved to: user_data/hyperopt_results/" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "📋 Next Steps:" -ForegroundColor Yellow
        Write-Host "  1. Review best parameters above"
        Write-Host "  2. Update strategy with optimal parameters"
        Write-Host "  3. Re-run backtest with new parameters"
        Write-Host "  4. Validate on different timerange"
        Write-Host ""
        Write-Host "💡 Tip: Save best parameters to a file" -ForegroundColor Cyan
        Write-Host "   freqtrade hyperopt-show -n 1 --print-json > optimal_params.json"
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "❌ Hyperopt failed!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host ""
    Write-Host "❌ Error: $_" -ForegroundColor Red
    exit 1
}
