# Backtesting Implementation Guide

## Overview

The backtesting system simulates trading decisions using historical data while accounting for real-world constraints and edge cases. This document outlines the technical implementation details and considerations.

## Core Components

### 1. Data Management

#### Historical Data Loading
```python
class HistoricalDataManager:
    def __init__(self, symbol: str, start_date: datetime, end_date: datetime):
        self.symbol = symbol
        self.start_date = start_date - timedelta(days=5)  # Extra days for warmup
        self.end_date = end_date
        self.data = None
        
    async def load_data(self):
        # Load 1-minute data for entire period
        self.data = await fetch_historical_data(
            symbol=self.symbol,
            start=self.start_date,
            end=self.end_date,
            interval="1m"
        )
        
    def get_snapshot(self, timestamp: datetime) -> Dict[str, pd.DataFrame]:
        """Get multi-timeframe snapshot as it would have appeared at timestamp."""
        # Ensure we only use data available at the timestamp
        mask = self.data.index <= timestamp
        available_data = self.data[mask]
        
        return {
            "1m": available_data.tail(100),
            "5m": resample_ohlcv(available_data, "5min").tail(100),
            "15m": resample_ohlcv(available_data, "15min").tail(100),
            "1h": resample_ohlcv(available_data, "1h").tail(100)
        }
```

#### Key Considerations:
1. Data Availability
   - Need extra historical data for warmup period
   - Must respect point-in-time knowledge (no future data leakage)
   - Handle missing data, gaps, and trading hours
   - Consider timezone conversions and market sessions

2. Data Quality
   - Validate OHLCV data consistency
   - Handle split/dividend adjustments
   - Account for tick size and price precision
   - Consider after-hours trading data

### 2. Position Management

#### Position Tracker
```python
class Position:
    def __init__(self, entry_price: float, size: float, entry_time: datetime,
                take_profit: float, stop_loss: float, confidence: float):
        self.entry_price = entry_price
        self.size = size  # Positive for long, negative for short
        self.entry_time = entry_time
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.confidence = confidence
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        
    def check_exit(self, current_price: float, current_time: datetime) -> bool:
        """Check if position should be exited based on price and time."""
        if self.size > 0:  # Long position
            if current_price >= self.take_profit:
                self.exit_price = self.take_profit
                self.exit_reason = "take_profit"
                return True
            if current_price <= self.stop_loss:
                self.exit_price = self.stop_loss
                self.exit_reason = "stop_loss"
                return True
        else:  # Short position
            if current_price <= self.take_profit:
                self.exit_price = self.take_profit
                self.exit_reason = "take_profit"
                return True
            if current_price >= self.stop_loss:
                self.exit_price = self.stop_loss
                self.exit_reason = "stop_loss"
                return True
                
        # Check other exit conditions (max duration, session end, etc.)
        return False
```

#### Key Considerations:
1. Entry Logic
   - Minimum confidence threshold
   - Maximum position size
   - Entry price slippage
   - Spread costs
   - Commission fees
   - Available margin

2. Exit Logic
   - Take-profit/stop-loss price slippage
   - Partial position exits
   - Time-based exits (session end, max duration)
   - Gap handling (opening gaps beyond stop-loss)
   - Market liquidity impact

### 3. Risk Management

```python
class RiskManager:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = 1.0
        self.max_daily_loss = initial_capital * 0.02  # 2% max daily loss
        self.positions = []
        self.daily_pnl = defaultdict(float)
        
    def can_enter_position(self, price: float, size: float, 
                          stop_loss: float) -> bool:
        # Check various risk constraints
        max_loss_per_trade = abs(price - stop_loss) * abs(size)
        if max_loss_per_trade > self.initial_capital * 0.01:  # 1% max loss per trade
            return False
            
        # Check daily loss limits
        today = datetime.now().date()
        if self.daily_pnl[today] < -self.max_daily_loss:
            return False
            
        return True
```

#### Key Considerations:
1. Position Sizing
   - Scale with confidence
   - Account for volatility
   - Consider correlation with existing positions
   - Respect margin requirements

2. Risk Limits
   - Maximum position size
   - Daily loss limits
   - Maximum number of concurrent positions
   - Maximum trades per day
   - Maximum drawdown threshold

### 4. Performance Analysis

```python
class PerformanceAnalyzer:
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        
    def add_trade(self, trade: Position):
        self.trades.append(trade)
        
    def calculate_metrics(self) -> Dict:
        return {
            "total_trades": len(self.trades),
            "win_rate": self._calculate_win_rate(),
            "profit_factor": self._calculate_profit_factor(),
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "max_drawdown": self._calculate_max_drawdown(),
            "avg_trade_duration": self._calculate_avg_duration(),
            "win_rate_by_hour": self._analyze_hourly_performance(),
            "win_rate_by_confidence": self._analyze_confidence_levels()
        }
```

#### Key Considerations:
1. Trade Analysis
   - Entry/exit efficiency
   - Slippage impact
   - Time decay of confidence
   - Win rate by various factors
   - Risk-adjusted returns

2. Visualization
   - Equity curve with drawdowns
   - Trade distribution
   - Performance by time of day
   - Performance by confidence level
   - Market condition correlation

## Implementation Challenges

### 1. API Cost Management
- LLM API calls are expensive
- Need efficient caching strategy
- Consider sampling frequency
- Batch processing for optimization

### 2. Performance Optimization
```python
class BacktestEngine:
    def __init__(self):
        self.cache = {}  # Cache for API responses
        self.batch_size = 100  # Process in batches
        
    async def run_parallel(self):
        """Run backtest with parallel processing."""
        # Split date range into batches
        batches = self._create_batches()
        
        # Process batches in parallel
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(self._process_batch(batch))
                for batch in batches
            ]
            
        # Combine results
        results = [task.result() for task in tasks]
        return self._merge_results(results)
```

### 3. Edge Cases
1. Market Events
   - Handle trading halts
   - Account for circuit breakers
   - Deal with extreme volatility
   - Handle corporate actions

2. Data Issues
   - Missing data points
   - Bad ticks/prices
   - After-hours trading
   - Holiday schedules

3. Position Management
   - Gap openings beyond stops
   - Partial fills
   - Position transfer across days
   - Multiple timeframe signals

## Future Enhancements

1. Advanced Features
   - Dynamic position sizing
   - Adaptive take-profit/stop-loss
   - Multi-asset correlation
   - Market regime detection

2. Optimization
   - Genetic algorithms for parameter tuning
   - Walk-forward analysis
   - Monte Carlo simulation
   - Machine learning integration

## Configuration Example

```python
backtest_config = {
    "capital": 100000,
    "position_sizing": {
        "max_size": 1.0,
        "size_by_confidence": True,
        "min_confidence": 0.6
    },
    "risk_management": {
        "max_daily_loss_pct": 2.0,
        "max_trade_loss_pct": 1.0,
        "max_drawdown_pct": 5.0
    },
    "timing": {
        "session_start": "09:30",
        "session_end": "16:00",
        "max_trade_duration": "4h",
        "min_time_between_trades": "5m"
    },
    "costs": {
        "commission_per_contract": 2.50,
        "slippage_factor": 0.0001
    }
}
``` 