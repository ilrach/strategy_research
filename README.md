# Strategy Research

Quantitative research framework for analyzing financial market strategies and portfolios.

## Project Structure

```
strategy_research/
├── strategies/                    # Strategy-specific research
│   ├── leveraged_etf_double_short/
│   │   ├── README.md             # Strategy overview and findings
│   │   ├── analysis/             # Core analysis scripts
│   │   ├── plots/                # Strategy visualizations
│   │   ├── data/                 # CSV outputs and cached data
│   │   └── archive/              # Deprecated/debug scripts
│   │
│   └── intraday_overnight/
│       ├── README.md
│       ├── analysis/
│       ├── plots/
│       ├── data/
│       └── archive/
│
├── fmp_client.py                 # FMP API client (shared utility)
├── .env                          # API keys (not committed)
├── .gitignore
└── README.md                     # This file
```

## Strategies

### [Leveraged ETF Double Short](strategies/leveraged_etf_double_short/)
**Status**: ✅ Viable with caveats

Short both bull and bear leveraged ETFs to capture volatility decay while earning interest on cash proceeds.

**Performance**: 9.3% CAGR, 3.6 Sharpe, -0.68% max drawdown (portfolio of 10 pairs)

**Key Insight**: 13/18 pairs have positive edge after borrow costs. TQQQ/SQQQ is best with 5.42 edge ratio.

### [Intraday vs Overnight Returns](strategies/intraday_overnight/)
**Status**: ⚠️ Research stage

Analysis of differential performance between market sessions (intraday vs overnight).

**Key Insight**: Overnight returns contribute majority of equity gains. Requires strategy development.

## Setup

1. **Install dependencies**:
   ```bash
   pip install requests pandas numpy matplotlib seaborn
   ```

2. **Configure API access**:
   Create a `.env` file in the project root:
   ```
   FMP_API_KEY=your_api_key_here
   ```

3. **Add to Python path** (for imports to work):
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/Users/bryn/projects/strategy_research"
   ```
   Or add to your shell profile for persistence.

## Usage

Navigate to a strategy directory and run analysis scripts:

```bash
# Example: Run leveraged ETF pair screening
cd strategies/leveraged_etf_double_short/analysis
python pair_screening.py

# Example: Run portfolio backtest
python backtest_portfolio.py
```

Each strategy directory contains:
- **README.md**: Strategy overview, findings, and script descriptions
- **analysis/**: Main analysis scripts
- **plots/**: Generated visualizations
- **data/**: CSV outputs and cached data
- **archive/**: Old/debug scripts

## Data Sources

- **FMP API**: Historical prices, dividend-adjusted data, fundamentals
- **iBorrowDesk API**: Historical borrow costs for short selling
- All analysis uses dividend-adjusted prices for accurate total returns

## Development Guidelines

### Adding a New Strategy

1. **Create directory structure**:
   ```bash
   mkdir -p strategies/[strategy_name]/{analysis,plots,data,archive}
   ```

2. **Create README.md** using this template:
   ```markdown
   # [Strategy Name]

   ## Overview
   Brief description and hypothesis

   ## Key Findings
   Performance metrics and insights

   ## Analysis Scripts
   List of scripts with descriptions

   ## Viability Assessment
   Is this strategy tradeable?
   ```

3. **Organize files**:
   - Core analysis → `analysis/`
   - Outputs/plots → `plots/`
   - Data files → `data/`
   - Debug/test → `archive/`

4. **Import the API client**:
   ```python
   from fmp_client import FMPClient
   ```

### Code Conventions

- Use descriptive file names (not version numbers)
- Document findings in strategy README
- Keep archive/ for debugging history
- Don't commit large data files or plots to git
- All scripts should be runnable from their directory

## Contributing

When researching a new strategy:

1. Create strategy directory under `strategies/`
2. Document hypothesis and approach in README
3. Build analysis scripts incrementally
4. Record findings and metrics in README
5. Keep archive of debugging work

Focus on:
- **Reproducibility**: Scripts should run without manual intervention
- **Documentation**: Explain what you learned, not just what you did
- **Organization**: Keep strategy work self-contained
