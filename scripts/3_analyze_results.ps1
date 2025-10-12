# Pragma Trading Bot - Results Analysis Script
# Analyzes backtest results and generates report

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pragma Trading Bot - Results Analysis" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Freqtrade is installed
$freqtradeInstalled = Get-Command freqtrade -ErrorAction SilentlyContinue
if (-not $freqtradeInstalled) {
    Write-Host "❌ ERROR: Freqtrade not found!" -ForegroundColor Red
    exit 1
}

# Check if results exist
$resultsDir = "user_data/backtest_results"
if (-not (Test-Path $resultsDir)) {
    Write-Host "❌ ERROR: No backtest results found!" -ForegroundColor Red
    Write-Host "Run backtest first: .\scripts\2_run_backtest.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "📊 Running analysis..." -ForegroundColor Cyan
Write-Host ""

# Run Freqtrade analysis
try {
    & freqtrade backtesting-analysis
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ Analysis Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    
    # Show latest results file
    $latestResult = Get-ChildItem $resultsDir -Filter "*.json" | 
        Sort-Object LastWriteTime -Descending | 
        Select-Object -First 1
    
    if ($latestResult) {
        Write-Host "📄 Latest result file:" -ForegroundColor Cyan
        Write-Host "  $($latestResult.FullName)" -ForegroundColor Gray
        Write-Host ""
    }
    
    Write-Host "💡 Key Metrics to Review:" -ForegroundColor Yellow
    Write-Host "  ✓ Total Profit %" -ForegroundColor White
    Write-Host "  ✓ Win Rate %" -ForegroundColor White
    Write-Host "  ✓ Sharpe Ratio (target > 1.5)" -ForegroundColor White
    Write-Host "  ✓ Max Drawdown % (target < 3%)" -ForegroundColor White
    Write-Host "  ✓ Profit Factor (target > 1.5)" -ForegroundColor White
    Write-Host "  ✓ Average Trade Duration" -ForegroundColor White
    Write-Host "  ✓ Number of Trades (target 10-20/day)" -ForegroundColor White
    Write-Host ""
    
    Write-Host "📋 Next Steps:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "If results are promising:" -ForegroundColor Green
    Write-Host "  1. Run hyperopt for optimization" -ForegroundColor White
    Write-Host "     .\scripts\4_run_hyperopt.ps1" -ForegroundColor Gray
    Write-Host ""
    Write-Host "If results need improvement:" -ForegroundColor Red
    Write-Host "  1. Review strategy parameters" -ForegroundColor White
    Write-Host "  2. Check regime detection accuracy" -ForegroundColor White
    Write-Host "  3. Analyze losing trades" -ForegroundColor White
    Write-Host "  4. Adjust confidence thresholds" -ForegroundColor White
    Write-Host ""
    
    # Generate summary report
    Write-Host "📝 Generating summary report..." -ForegroundColor Cyan
    
    $reportPath = "docs/BACKTEST_RESULTS.md"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    $report = @"
# Backtest Results Summary

**Generated:** $timestamp

## Configuration

- **Strategy:** RegimeAdaptiveStrategy
- **Timerange:** See results above
- **Pairs:** Multiple (BTC/USDT, ETH/USDT, etc.)
- **Timeframe:** 5m

## Results

[Copy key metrics from console output above]

### Key Metrics

- **Total Profit:** [Fill in]
- **Win Rate:** [Fill in]
- **Sharpe Ratio:** [Fill in]
- **Max Drawdown:** [Fill in]
- **Profit Factor:** [Fill in]
- **Total Trades:** [Fill in]

### Targets Comparison

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Sharpe Ratio | > 1.5 | [Fill] | [✅/❌] |
| Max Drawdown | < 3% | [Fill] | [✅/❌] |
| Win Rate | > 70% | [Fill] | [✅/❌] |
| Profit Factor | > 1.5 | [Fill] | [✅/❌] |

## Regime Performance

[Analyze performance by regime if available]

### Trending Regime
- Trades: [Fill]
- Win Rate: [Fill]
- Avg Profit: [Fill]

### Low Volatility Regime
- Trades: [Fill]
- Win Rate: [Fill]
- Avg Profit: [Fill]

### High Volatility Regime
- Trades: [Fill]
- Win Rate: [Fill]
- Avg Profit: [Fill]

## Analysis

### Strengths
- [List what worked well]

### Weaknesses
- [List areas for improvement]

### Observations
- [Key findings]

## Recommendations

### Immediate Actions
1. [Action items]

### Hyperopt Focus
- Parameters to optimize: [List]
- Expected improvements: [Describe]

## Next Steps

- [ ] Review and fill in metrics above
- [ ] Analyze regime-specific performance
- [ ] Decide: Continue to hyperopt OR Improve strategy
- [ ] Document any changes made

---

**Note:** This is a template. Fill in actual values from backtest results.
"@

    $report | Out-File $reportPath -Encoding UTF8
    
    Write-Host "✅ Report template created: $reportPath" -ForegroundColor Green
    Write-Host "   Please fill in the actual values!" -ForegroundColor Yellow
    Write-Host ""
    
} catch {
    Write-Host ""
    Write-Host "❌ Error: $_" -ForegroundColor Red
    exit 1
}
