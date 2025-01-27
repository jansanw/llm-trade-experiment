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

### Running the Live Bot
```bash
python src/main.py --mode live --symbol MNQ
```

### Running Backtests
```bash
python src/main.py --mode backtest --symbol MNQ --start-date 2024-01-01 --end-date 2024-02-01
```

### Starting the Dashboard
```bash
python src/dashboard.py
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