# Strategy Research

Quantitative research framework for analyzing financial market strategies and portfolios.

## Project Structure

```
strategy_research/
├── fmp_client.py       # Financial Modeling Prep API client for market data
├── research/           # Primary research scripts and analysis modules
├── misc_tests/         # Experimental and ad-hoc analysis scripts
├── plots/              # Generated charts and visualizations (timestamped)
├── logs/               # Script execution logs (timestamped)
├── risk_metrics/       # Risk and performance metric outputs
└── .env                # API keys and configuration (not committed)
```

## Setup

1. Install dependencies:
   ```bash
   pip install requests pandas numpy matplotlib
   ```

2. Configure API access:
   - Create a `.env` file in the project root
   - Add your Financial Modeling Prep API key:
     ```
     FMP_API_KEY=your_api_key_here
     ```

## Usage

Research scripts can be run directly from the command line:

```bash
# Run analysis scripts
python research/script_name.py [arguments]
python misc_tests/script_name.py [arguments]
```

Scripts automatically generate:
- **Plots**: Saved to `plots/` with timestamp and descriptive names
- **Logs**: Execution output saved to `logs/` with matching timestamps
- **Risk Metrics**: Performance statistics saved to `risk_metrics/`

## Data Sources

- **FMP API**: Historical price data, dividend-adjusted prices, and fundamental data
- All analysis uses dividend-adjusted prices for accurate total return calculations

## Output Conventions

All generated files follow a consistent naming pattern:
```
YYYYMMDD_HHMMSS_description.ext
```

This ensures chronological ordering and prevents file overwrites.

## Development

- Add new research to `research/` for production analysis
- Use `misc_tests/` for experimental or one-off analysis
- Keep API keys in `.env` (already in `.gitignore`)
- Commit code and documentation, not generated outputs
