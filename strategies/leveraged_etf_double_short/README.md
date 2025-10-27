# Leveraged ETF Double Short Strategy

## Overview

Strategy that shorts both the bull and bear ETF in leveraged pairs to capture volatility decay while earning interest on cash proceeds.

**Core Hypothesis**: Leveraged ETFs experience volatility decay over time due to daily rebalancing. By shorting both sides of a pair, we create a market-neutral position that benefits from this decay while earning interest on the short proceeds.

## Key Findings

### Performance Metrics (3-Year Backtest, 2022-2025)

#### Single Pair (TQQQ/SQQQ)
| Metric | Without Borrow Costs | With Borrow Costs |
|--------|---------------------|-------------------|
| Total Return | +23.51% | +18.97% |
| CAGR | 10.83% | 8.83% |
| Volatility | 0.84% | 0.84% |
| Sharpe Ratio | 8.46 | 6.97 |
| Max Drawdown | -0.20% | -0.22% |
| Win Rate | - | 67.3% |

**Borrow Costs**: TQQQ 0.69% + SQQQ 1.83% = 2.52% annually

#### Portfolio (Top 10 Volatile Pairs)
| Metric | Value |
|--------|-------|
| Total Return | +20.02% |
| CAGR | 9.31% |
| Volatility | 1.70% |
| Sharpe Ratio | 3.62 |
| Max Drawdown | -0.68% |
| Win Rate | 58.2% |

**Configuration**: 10 pairs at 20% each (10% bull + 10% bear), 200% cash earning BIL returns

### Edge Analysis

**Edge Threshold**: For each 1% borrow cost, need 4.5% volatility for positive edge

**Top Pairs by Edge Ratio**:
1. TQQQ/SQQQ - Edge: 5.42 (Vol: 61.5%, Borrow: 2.5%)
2. SOXL/SOXS - Edge: 3.66 (Vol: 105.4%, Borrow: 6.4%)
3. YINN/YANG - Edge: 3.27 (Vol: 92.5%, Borrow: 6.3%)
4. UPRO/SPXU - Edge: 3.19 (Vol: 46.7%, Borrow: 3.3%)
5. SPXL/SPXS - Edge: 2.67 (Vol: 46.6%, Borrow: 3.9%)

**Avoid**: USD/SSG (SSG 44% borrow cost), TMF/TMV (TMV 21% borrow cost)

### Volatility Regime Analysis

Strategy performs modestly better in higher volatility environments:
- Low realized vol (deciles 1-3): 6.43% annualized
- High realized vol (deciles 8-10): 7.40% annualized
- **+0.97% edge in high volatility**

## Analysis Scripts

### Core Analysis
- `analysis/pair_screening.py` - Screen all leveraged ETF pairs for volatility and volume
- `analysis/backtest_single_pair.py` - Backtest strategy on single pair (e.g., TQQQ/SQQQ)
- `analysis/backtest_portfolio.py` - Backtest portfolio of top 10 pairs
- `analysis/backtest_with_costs.py` - Single pair backtest including borrow costs
- `analysis/borrow_cost_analysis.py` - Analyze volatility vs borrow cost edge ratio
- `analysis/volatility_regime_analysis.py` - Performance by SPY realized volatility decile
- `analysis/cost_impact_analysis.py` - Compare returns with/without borrow costs
- `analysis/fetch_borrow_costs.py` - Fetch current borrow costs from iBorrowDesk API
- `analysis/plot_results.py` - Generate visualizations

### Archive (Development/Debug)
- `archive/` - Contains debug scripts, test scripts, and deprecated versions

## Data Files

### Results
- `data/pair_volatility_borrow_analysis.csv` - Complete pair analysis with edge ratios
- `data/portfolio_backtest_v2_summary.csv` - Portfolio backtest metrics
- `data/portfolio_backtest_v2_equity.csv` - Portfolio equity curve
- `data/backtest_with_borrow_costs_results.csv` - Single pair results with costs
- `data/rvol_decile_stats.csv` - Performance by volatility regime

### Borrow Costs
- `data/borrow_costs/TQQQ_borrow_costs.csv` - Historical TQQQ borrow fees
- `data/borrow_costs/SQQQ_borrow_costs.csv` - Historical SQQQ borrow fees

## Strategy Mechanics

### Position Structure (Single Pair)
- Start with $100k capital
- Short $50k bull ETF, short $50k bear ETF
- Hold $200k cash earning BIL (T-bill) returns
- Daily return = 2×BIL - 0.5×Bull - 0.5×Bear - Borrow_Costs

### Position Structure (Portfolio)
- 10 pairs at 20% each (10% bull + 10% bear)
- Total: 200% shorts → 200% cash
- Weekly rebalancing to maintain weights

### Risk Management
- Very low volatility (~1% for portfolio)
- Minimal drawdowns (<1%)
- Market-neutral (no directional exposure)
- Main risk: sustained convergence of bull/bear (unlikely)

## Important Notes

### Data Sources
- Price data: Financial Modeling Prep (FMP) API
- Borrow costs: iBorrowDesk API (limited historical data)
- Analysis period: 3 years (2022-2025)

### Limitations
1. **Borrow costs**: Many pairs lack historical borrow cost data
2. **Execution**: Real-world execution likely has higher costs than modeled
3. **Borrow availability**: High-borrow pairs may have limited availability
4. **Date ordering bug**: Early backtests had reversed date ordering (fixed)
5. **Backtest period**: Only 3 years, includes both bull and volatile markets

### Critical Bug Fixes
- **2025-10-27**: Fixed date ordering bug in FMP data (was descending, caused reversed backtest)
- **2025-10-27**: Fixed cash multiplier in portfolio backtest (was using wrong leverage)

## Viability Assessment

✅ **Viable Strategy** with caveats:

**Strengths**:
- Consistent positive returns across different pairs
- Very low volatility and drawdowns
- Excellent risk-adjusted returns (Sharpe 3-7)
- Works better in high volatility environments
- 13 out of 18 pairs have positive edge after costs

**Risks**:
- Borrow costs are significant drag (2-10% annually)
- Limited historical borrow data for validation
- Execution complexity (shorting, rebalancing)
- Regulatory/broker constraints on short selling
- Requires substantial capital for diversification

**Recommended Approach**:
- Start with TQQQ/SQQQ (best edge ratio, liquid, lower costs)
- Monitor borrow costs closely (can spike)
- Diversify across top 5-7 pairs with positive edge
- Avoid pairs with extreme borrow costs (>10% total)

## Future Research Directions

1. **Longer backtest**: Extend to 10+ years if borrow data available
2. **Transaction costs**: Model realistic execution costs
3. **Dynamic allocation**: Weight by edge ratio instead of equal weight
4. **Borrow cost hedging**: Strategies to mitigate cost spikes
5. **Alternative pairs**: Explore 2x leveraged pairs (lower vol, lower costs)
6. **Rebalancing optimization**: Test different frequencies and thresholds
