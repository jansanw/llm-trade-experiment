# LLM Trading Bot

A trading bot that uses Large Language Models (LLM) to make trading decisions based on multi-timeframe market data. The system supports multiple LLM providers (DeepSeek, OpenAI, Claude, local models) and includes backtesting capabilities.

## Features

- Multi-timeframe analysis (1min, 5min, 15min, 1hour candles)
- Flexible LLM provider system (DeepSeek, OpenAI, Claude, local models)
- Real-time market data visualization
- Comprehensive backtesting engine
- Support for multiple assets (MNQ futures, crypto, SPY)
- Web-based dashboard for monitoring and analysis

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   DEEPSEEK_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   ```

## Usage

The bot can be run in several modes:

```bash

# Single analysis
python -m src.main --mode oneshot --symbol QQQ --prompt-type fvg
python -m src.main --mode oneshot --symbol QQQ --prompt-type raw

# Live trading mode
python -m src.main --mode live --symbol SPY

# Backtest mode
python -m src.main --mode backtest --symbol QQQ --start-date 2025-01-12 --end-date 2025-01-26 --provider deepseek

# Dashboard mode (default)
python -m src.main --mode dashboard --symbol SPY
```

### Prompt Types

The bot supports different types of market analysis prompts:

```bash
# Use Fair Value Gap (FVG) analysis (default)
python -m src.main --mode oneshot --symbol SPY --prompt-type fvg

# Use basic OHLCV analysis
python -m src.main --mode oneshot --symbol SPY --prompt-type v0
```

### Dry Run Mode

You can test the prompt generation without making API calls using the `--dry-run` flag:

```bash
# Test FVG analysis prompt
python -m src.main --mode oneshot --symbol SPY --prompt-type fvg --dry-run

# Test basic OHLCV analysis prompt
python -m src.main --mode oneshot --symbol SPY --prompt-type v0 --dry-run
```

This is useful for:
- Debugging prompt generation
- Testing data processing
- Avoiding API costs during development
- Validating the data pipeline

## Backtesting

The bot now includes take-profit and stop-loss levels in its trading decisions. Here's how to effectively backtest strategies:

### Basic Backtest

Run a simple backtest over a date range:

```bash
python -m src.main --mode backtest --symbol SPY \
    --start-date 2024-01-01 --end-date 2024-01-31 \
    --prompt-type fvg  # or v0
```

### Backtesting Strategy

The backtesting process:

1. **Data Collection**
   - Historical minute data is fetched for the specified period
   - Data is organized into multiple timeframes (1min, 5min, 15min, 1hour)
   - Each decision point has access to the last 100 candles of each timeframe

2. **Trading Logic**
   - For each minute:
     - Generate trading decision with position, confidence, take-profit, and stop-loss
     - Enter position if confidence exceeds threshold
     - Exit position if:
       - Take-profit level is reached
       - Stop-loss level is reached
       - Position has been held for maximum duration

3. **Performance Metrics**
   - Win rate
   - Profit factor
   - Maximum drawdown
   - Sharpe ratio
   - Average trade duration
   - Number of trades
   - Distribution of profits/losses

### Optimization Ideas

1. **Entry Filters**
   - Minimum confidence threshold
   - Time-of-day restrictions
   - Minimum volume requirements
   - Maximum spread limits

2. **Exit Strategies**
   - Trailing stop-loss
   - Time-based exits
   - Partial profit taking
   - Break-even stops

3. **Position Sizing**
   - Scale position size with confidence
   - Account for volatility
   - Consider recent trade outcomes

4. **Risk Management**
   - Maximum position size
   - Daily loss limits
   - Maximum trades per day
   - Correlation with market conditions

### Example Backtest Configurations

Test different prompt types:
```bash
# Compare FVG vs Basic analysis
python -m src.main --mode backtest --symbol SPY \
    --start-date 2024-01-01 --end-date 2024-01-31 \
    --prompt-type fvg

python -m src.main --mode backtest --symbol SPY \
    --start-date 2024-01-01 --end-date 2024-01-31 \
    --prompt-type v0
```

Test different time periods:
```bash
# Morning session only
python -m src.main --mode backtest --symbol SPY \
    --start-date 2024-01-01 --end-date 2024-01-31 \
    --session-start 09:30 --session-end 11:30

# Afternoon session only
python -m src.main --mode backtest --symbol SPY \
    --start-date 2024-01-01 --end-date 2024-01-31 \
    --session-start 13:30 --session-end 16:00
```

### Analyzing Results

The backtest results include:

1. **Trade Log**
   - Entry/exit times and prices
   - Take-profit and stop-loss levels
   - Actual profit/loss
   - Trade duration
   - Model confidence
   - Reasoning for entry

2. **Performance Summary**
   - Overall P&L
   - Win rate by:
     - Time of day
     - Day of week
     - Confidence level
     - Prompt type
   - Average profit vs loss
   - Maximum consecutive wins/losses

3. **Visualization**
   - Equity curve
   - Drawdown chart
   - Trade distribution
   - Daily/weekly P&L
   - Win rate heatmap

## Project Structure

- `src/`
  - `llm/` - LLM provider wrappers
  - `data/` - Market data fetching and processing
  - `bot/` - Core trading bot logic
  - `backtest/` - Backtesting engine
  - `dashboard/` - Web visualization interface
  - `config/` - Configuration files
  - `utils/` - Utility functions 