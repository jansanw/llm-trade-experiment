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
# Live trading mode
python -m src.main --mode live --symbol SPY

# Backtest mode
python -m src.main --mode backtest --symbol SPY --start-date 2024-01-01 --end-date 2024-01-31

# Single analysis
python -m src.main --mode oneshot --symbol SPY

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

## Project Structure

- `src/`
  - `llm/` - LLM provider wrappers
  - `data/` - Market data fetching and processing
  - `bot/` - Core trading bot logic
  - `backtest/` - Backtesting engine
  - `dashboard/` - Web visualization interface
  - `config/` - Configuration files
  - `utils/` - Utility functions 