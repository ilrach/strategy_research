# Intraday vs Overnight Returns Strategy

## Overview

Analysis of the differential performance between intraday (market open to close) and overnight (close to next open) returns for various equity indices and ETFs.

**Core Hypothesis**: Equity returns are not uniformly distributed throughout the day. Overnight returns (when markets are closed) may exhibit different risk/return characteristics than intraday returns.

## Key Findings

### SPY Analysis
- **Overnight returns** have historically contributed the majority of total returns
- **Intraday returns** show different volatility patterns
- Strong asymmetry in return attribution between sessions

### BTAL/SPY Overnight Analysis
Analysis of BTAL (AGF US Market Neutral Anti-Beta Fund) versus SPY overnight returns:
- BTAL designed to have negative beta to equities
- Overnight period analysis reveals correlation patterns
- Beta hedging effectiveness varies by market regime

## Analysis Scripts

### Core Analysis
- `analysis/intraday_overnight_combined.py` - Main analysis comparing intraday vs overnight for multiple tickers
- `analysis/btal_spy_analysis.py` - Detailed BTAL vs SPY overnight return analysis
- `analysis/btal_spy_overnight_beta.py` - Beta analysis for BTAL overnight returns
- `analysis/weighted_portfolio.py` - Portfolio construction using intraday/overnight insights

## Data Files

Currently no CSV outputs (analysis focuses on visualizations)

## Strategy Mechanics

### Session Definitions
- **Overnight**: Previous close to current open
- **Intraday**: Current open to current close

### Analysis Approach
1. Fetch historical price data (open, close)
2. Calculate session-specific returns
3. Compare cumulative performance
4. Analyze risk metrics by session
5. Test portfolio combinations

## Important Notes

### Data Sources
- Historical OHLC data from FMP API
- Dividend-adjusted prices used for accurate returns
- Multiple tickers analyzed: SPY, QQQ, BTAL

### Limitations
1. Does not account for execution costs of session-based trading
2. Gap risk in overnight positions
3. Limited to US market hours definition
4. Does not model actual trading implementation

## Viability Assessment

⚠️ **Research Stage** - Requires further development:

**Insights Gained**:
- Clear documentation of overnight vs intraday return patterns
- BTAL behavior analysis provides anti-beta insights
- Foundation for potential session-based strategies

**Next Steps Needed**:
1. Quantify the overnight premium magnitude and consistency
2. Develop tradeable strategy based on insights
3. Model realistic execution for session-based entries/exits
4. Backtest complete strategy with costs
5. Risk management for gap exposure

## Future Research Directions

1. **Strategy development**: Convert insights into actionable trading rules
2. **Multiple assets**: Expand analysis to more tickers
3. **Regime analysis**: Performance by market conditions
4. **Portfolio optimization**: Optimal allocation between session exposures
5. **Cost modeling**: Realistic transaction costs for implementation
6. **Liquidity analysis**: Impact of trading at open/close
