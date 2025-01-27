from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from ..bot.trading_bot import TradingBot
from ..data.market_data import MarketDataFetcher
import asyncio
import pytz
from tabulate import tabulate
import logging
import traceback

class Trade:
    """Container for trade information."""
    def __init__(
        self,
        entry_time: datetime,
        position: float,
        entry_price: float,
        take_profit: Optional[float],
        stop_loss: Optional[float],
        confidence: float,
        reasoning: str
    ):
        self.entry_time = entry_time
        self.position = position  # 1.0 for long, -1.0 for short
        self.entry_price = entry_price
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.confidence = confidence
        self.reasoning = reasoning
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.status = "OPEN"
        
    def close(self, exit_time: datetime, exit_price: float, status: str = "CLOSED"):
        """Close the trade and calculate P&L."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.status = status
        
        if self.entry_price and self.exit_price:
            price_diff = self.exit_price - self.entry_price
            self.pnl = price_diff * self.position
            if self.entry_price != 0:
                self.pnl_pct = (self.pnl / abs(self.entry_price)) * 100
        
    def __str__(self):
        status_str = f"{self.status}"
        if self.status == "CLOSED":
            status_str += f" ({'+' if self.pnl >= 0 else ''}{self.pnl:.2f} / {self.pnl_pct:.1f}%)"
        return f"{self.entry_time.strftime('%Y-%m-%d %H:%M')} {status_str} | {self.position:+.1f} @ {self.entry_price:.2f}"

    def to_dict(self) -> Dict:
        """Convert trade to dictionary for results."""
        return {
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "position": "LONG" if self.position > 0 else "SHORT",
            "size": abs(self.position),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "status": self.status,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }

class BacktestEngine:
    """Engine for running historical backtests."""
    
    def __init__(self, bot: TradingBot, start_date: datetime, end_date: datetime):
        """Initialize the backtest engine.
        
        Args:
            bot: TradingBot instance
            start_date: Start date (datetime or str in YYYY-MM-DD format)
            end_date: End date (datetime or str in YYYY-MM-DD format)
        """
        self.bot = bot
        
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            self.start_date = start_date
            
        if isinstance(end_date, str):
            self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            self.end_date = end_date
            
        self.trades: List[Trade] = []
        self.target_time = time(11, 5)  # 11:05 AM
        self.et_tz = pytz.timezone('US/Eastern')
        self.logger = logging.getLogger(__name__)
        
    def _is_market_day(self, date: datetime) -> bool:
        """Check if date is a weekday."""
        return date.weekday() < 5  # Monday = 0, Friday = 4
        
    def _get_target_times(self) -> List[datetime]:
        """Get list of target timestamps (11:05 AM ET each market day)."""
        self.logger.info("Generating target timestamps...")
        times = []
        current = self.start_date
        
        while current <= self.end_date:
            if self._is_market_day(current):
                # Create timestamp for 11:05 AM ET on this day
                target = datetime.combine(current.date(), self.target_time)
                target = self.et_tz.localize(target)
                times.append(target)
                self.logger.debug(f"Added target time: {target}")
            current += timedelta(days=1)
            
        self.logger.info(f"Generated {len(times)} target timestamps")
        return times
        
    async def run(self) -> List[Trade]:
        """Run the backtest over the specified period."""
        self.logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        
        # Generate target timestamps
        target_times = self._get_target_times()
        self.logger.info(f"Generated {len(target_times)} target timestamps")
        
        print("\nStarting backtest...")
        print(f"Running on {len(target_times)} market days at 11:05 AM ET\n")
        
        trades = []
        active_trade = None
        
        for timestamp in target_times:
            self.logger.info(f"\nProcessing timestamp: {timestamp}")
            
            try:
                # Get trading decision
                self.logger.info("Fetching trading decision...")
                decision = await self.bot.get_trading_decision(timestamp)
                self.logger.info(f"Got decision: {decision}")
                
                # Get current price from minute data
                self.logger.info("Fetching minute data for current price...")
                minute_data = await self.bot.get_minute_data(timestamp, timestamp + timedelta(minutes=1))
                if minute_data.empty:
                    self.logger.warning(f"No minute data available for {timestamp}")
                    continue
                    
                current_price = minute_data.iloc[-1]["close"]
                self.logger.info(f"Current price: {current_price}")
                
                # Close active trade if it exists
                if active_trade:
                    self.logger.info(f"Processing active trade: {active_trade}")
                    # Get minute data since trade entry
                    self.logger.info(f"Fetching minute data from {active_trade.entry_time} to {timestamp}")
                    trade_data = await self.bot.get_minute_data(
                        active_trade.entry_time, 
                        timestamp + timedelta(minutes=1)
                    )
                    
                    if not trade_data.empty:
                        self.logger.info(f"Got {len(trade_data)} bars of minute data")
                        # Check for take-profit or stop-loss hits
                        for _, bar in trade_data.iterrows():
                            if active_trade.take_profit and active_trade.stop_loss:
                                if active_trade.position > 0:  # Long position
                                    if bar["high"] >= active_trade.take_profit:
                                        self.logger.info("Take-profit hit for long position")
                                        active_trade.close(bar.name, active_trade.take_profit, "TP_HIT")
                                        trades.append(active_trade)
                                        active_trade = None
                                        break
                                    elif bar["low"] <= active_trade.stop_loss:
                                        self.logger.info("Stop-loss hit for long position")
                                        active_trade.close(bar.name, active_trade.stop_loss, "SL_HIT")
                                        trades.append(active_trade)
                                        active_trade = None
                                        break
                                else:  # Short position
                                    if bar["low"] <= active_trade.take_profit:
                                        self.logger.info("Take-profit hit for short position")
                                        active_trade.close(bar.name, active_trade.take_profit, "TP_HIT")
                                        trades.append(active_trade)
                                        active_trade = None
                                        break
                                    elif bar["high"] >= active_trade.stop_loss:
                                        self.logger.info("Stop-loss hit for short position")
                                        active_trade.close(bar.name, active_trade.stop_loss, "SL_HIT")
                                        trades.append(active_trade)
                                        active_trade = None
                                        break
                    else:
                        self.logger.warning("No trade data available for active trade")
                    
                    # Close trade at current price if still open
                    if active_trade:
                        self.logger.info(f"Closing trade at current price: {current_price}")
                        active_trade.close(timestamp, current_price)
                        trades.append(active_trade)
                        active_trade = None
                
                # Open new trade if decision is not neutral
                if abs(decision.get("position", 0)) > 0 and decision.get("confidence", 0) > 0:
                    if decision.get("take_profit") and decision.get("stop_loss"):
                        self.logger.info("Opening new trade...")
                        active_trade = Trade(
                            entry_time=timestamp,
                            position=1.0 if decision["position"] > 0 else -1.0,  # Normalize to 1.0/-1.0
                            entry_price=current_price,
                            take_profit=decision["take_profit"],
                            stop_loss=decision["stop_loss"],
                            confidence=decision["confidence"],
                            reasoning=decision.get("reasoning", "")
                        )
                        self.logger.info(f"Opened new trade: {active_trade}")
                
                self.logger.info(f"Processed {timestamp} successfully")
                
            except Exception as e:
                self.logger.error(f"Error processing {timestamp}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Close any remaining open trade
        if active_trade:
            self.logger.info("Closing final active trade")
            active_trade.close(target_times[-1], current_price)
            trades.append(active_trade)
        
        # Calculate and print results
        self._print_results(trades)
        
        return trades

    def _print_results(self, trades):
        """Print backtest results in a nice format."""
        if not trades:
            print("\nNo trades executed during backtest period.")
            return
            
        # Calculate statistics
        total_trades = len(trades)
        closed_trades = [t for t in trades if t.status != "OPEN"]
        winning_trades = len([t for t in closed_trades if t.pnl > 0])
        tp_hits = len([t for t in closed_trades if t.status == "TP_HIT"])
        sl_hits = len([t for t in closed_trades if t.status == "SL_HIT"])
        total_pnl = sum(t.pnl for t in closed_trades)
        avg_pnl_pct = sum(t.pnl_pct for t in closed_trades) / len(closed_trades) if closed_trades else 0
        
        # Print trade list
        print("\nTrade List:")
        print("-" * 100)
        headers = ["Date", "Position", "Entry", "Exit", "TP", "SL", "P&L", "Status", "Conf"]
        rows = []
        for trade in trades:
            rows.append([
                trade.entry_time.strftime("%Y-%m-%d %H:%M"),
                "LONG" if trade.position > 0 else "SHORT",
                f"{trade.entry_price:.2f}",
                f"{trade.exit_price:.2f}" if trade.exit_price else "-",
                f"{trade.take_profit:.2f}" if trade.take_profit else "-",
                f"{trade.stop_loss:.2f}" if trade.stop_loss else "-",
                f"{trade.pnl:+.2f} ({trade.pnl_pct:+.1f}%)" if trade.status != "OPEN" else "-",
                trade.status,
                f"{trade.confidence:.1%}"
            ])
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        # Print summary statistics
        print("\nPerformance Summary:")
        print("-" * 40)
        print(f"Total Trades: {total_trades}")
        if closed_trades:
            win_rate = (winning_trades / len(closed_trades)) * 100
            print(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")
            print(f"Take-Profit Hits: {tp_hits} ({(tp_hits/len(closed_trades))*100:.1f}%)")
            print(f"Stop-Loss Hits: {sl_hits} ({(sl_hits/len(closed_trades))*100:.1f}%)")
            print(f"Total P&L: ${total_pnl:.2f}")
            print(f"Average P&L per Trade: {avg_pnl_pct:.1f}%")
            print(f"Max Drawdown: ${min(t.pnl for t in closed_trades):.2f}")
            print(f"Best Trade: ${max(t.pnl for t in closed_trades):.2f}") 