from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from ..bot.trading_bot import TradingBot
from ..data.market_data import MarketDataFetcher

class BacktestResult:
    """Container for backtest results."""
    
    def __init__(
        self,
        trades: List[Dict],
        equity_curve: pd.DataFrame,
        performance_metrics: Dict[str, float]
    ):
        self.trades = trades
        self.equity_curve = equity_curve
        self.performance_metrics = performance_metrics
        
    def __str__(self) -> str:
        """String representation of backtest results."""
        metrics = [f"{k}: {v:.2f}" for k, v in self.performance_metrics.items()]
        return "\n".join([
            "Backtest Results:",
            "---------------",
            f"Total Trades: {len(self.trades)}",
            "Performance Metrics:",
            *metrics
        ])

class BacktestEngine:
    """Engine for running historical backtests."""
    
    def __init__(
        self,
        bot: TradingBot,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.0001  # 0.01%
    ):
        """
        Initialize the backtest engine.
        
        Args:
            bot: TradingBot instance to test
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Starting capital
            commission_rate: Commission rate per trade
        """
        self.bot = bot
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        
        self.trades = []
        self.positions = []
        self.equity_curve = []
        
    async def run(self) -> BacktestResult:
        """Run the backtest."""
        current_time = self.start_date
        current_capital = self.initial_capital
        current_position = 0.0
        
        while current_time <= self.end_date:
            # Skip non-trading hours (assuming market hours 9:30-16:00 EST)
            if not self._is_trading_hour(current_time):
                current_time += timedelta(minutes=1)
                continue
                
            # Get bot's decision for this point in time
            decision = await self.bot.get_trading_decision(current_time)
            
            # Check if we should take action
            if (abs(decision["position"]) >= self.bot.position_threshold and
                decision["confidence"] >= self.bot.min_confidence):
                
                # Calculate trade details
                new_position = decision["position"]
                position_change = new_position - current_position
                
                if abs(position_change) > 0:
                    # Get current price
                    _, _, _, min1_df = await self.bot.provider.fetch_multi_timeframe_data(
                        self.bot.symbol,
                        current_time
                    )
                    price = min1_df.iloc[-1]["close"]
                    
                    # Calculate trade impact
                    trade_value = abs(position_change) * current_capital
                    commission = trade_value * self.commission_rate
                    
                    # Record trade
                    trade = {
                        "timestamp": current_time,
                        "action": "BUY" if position_change > 0 else "SELL",
                        "price": price,
                        "size": abs(position_change),
                        "value": trade_value,
                        "commission": commission,
                        "confidence": decision["confidence"],
                        "reasoning": decision["reasoning"]
                    }
                    self.trades.append(trade)
                    
                    # Update position and capital
                    current_position = new_position
                    current_capital -= commission
                    
            # Record current equity
            self.positions.append({
                "timestamp": current_time,
                "position": current_position,
                "capital": current_capital
            })
            
            # Move to next minute
            current_time += timedelta(minutes=1)
            
        # Calculate performance metrics
        return self._calculate_results()
    
    def _is_trading_hour(self, dt: datetime) -> bool:
        """Check if given datetime is during trading hours."""
        # Convert to EST
        hour = dt.hour
        minute = dt.minute
        
        # Trading hours 9:30 AM - 4:00 PM EST
        return (
            (hour == 9 and minute >= 30) or
            (hour > 9 and hour < 16) or
            (hour == 16 and minute == 0)
        )
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results and performance metrics."""
        # Create equity curve
        equity_df = pd.DataFrame(self.positions)
        
        # Calculate returns
        equity_df["returns"] = equity_df["capital"].pct_change()
        
        # Calculate metrics
        total_return = (equity_df["capital"].iloc[-1] / self.initial_capital) - 1
        
        # Annualized return (assuming 252 trading days)
        days = (self.end_date - self.start_date).days
        ann_return = ((1 + total_return) ** (252 / days)) - 1
        
        # Sharpe ratio (assuming risk-free rate of 0.02)
        rf_daily = 0.02 / 252
        excess_returns = equity_df["returns"] - rf_daily
        sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        
        # Maximum drawdown
        rolling_max = equity_df["capital"].expanding().max()
        drawdowns = equity_df["capital"] / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        metrics = {
            "total_return": total_return * 100,  # as percentage
            "annualized_return": ann_return * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown * 100,
            "total_trades": len(self.trades),
            "win_rate": self._calculate_win_rate() * 100
        }
        
        return BacktestResult(self.trades, equity_df, metrics)
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trades."""
        if not self.trades:
            return 0.0
            
        profitable_trades = sum(
            1 for t in self.trades
            if (t["action"] == "BUY" and t["price"] < self.trades[-1]["price"]) or
               (t["action"] == "SELL" and t["price"] > self.trades[-1]["price"])
        )
        
        return profitable_trades / len(self.trades) 