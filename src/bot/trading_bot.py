from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import asyncio
import logging
from src.llm.base import LLMProvider
from src.data.market_data import MarketDataFetcher, MarketDataProvider

class TradingBot:
    """Main trading bot class that coordinates LLM decisions with market data."""
    
    def __init__(self, symbol: str, data_fetcher, llm):
        """Initialize the trading bot.
        
        Args:
            symbol: Trading symbol (e.g. SPY, BTC-USD)
            data_fetcher: Market data fetcher instance
            llm: LLM provider instance
        """
        self.symbol = symbol
        self.data_fetcher = data_fetcher
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        
    async def get_trading_decision(self, timestamp=None):
        """Get trading decision for the current market state.
        
        Args:
            timestamp: Optional timestamp to get decision for (used in backtesting)
            
        Returns:
            dict: Trading decision with position, confidence, take-profit, stop-loss and reasoning
        """
        try:
            # Fetch multi-timeframe data
            hourly_df, min15_df, min5_df, min1_df = await self.data_fetcher.fetch_multi_timeframe_data(end_time=timestamp)
            
            # Get trading decision from LLM
            decision = await self.llm.get_trading_decision(
                hourly_df=hourly_df,
                min15_df=min15_df,
                min5_df=min5_df,
                min1_df=min1_df
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error getting trading decision: {str(e)}")
            return {
                "position": 0.0,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }
            
    async def get_minute_data(self, start_time, end_time):
        """Get 1-minute candle data between specified timestamps.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            pandas.DataFrame: 1-minute OHLCV data
        """
        try:
            return await self.data_fetcher.get_candles(
                symbol=self.symbol,
                interval="1m",
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            self.logger.error(f"Error fetching minute data: {str(e)}")
            return pd.DataFrame()
            
    async def run_live(self):
        """Run the bot in live trading mode."""
        self.logger.info(f"Starting live trading for {self.symbol}")
        
        while True:
            try:
                decision = await self.get_trading_decision()
                
                # Log the decision
                self.logger.info(
                    f"Decision: pos={decision['position']:.2f}, "
                    f"conf={decision['confidence']:.2f}, "
                    f"reason={decision['reasoning']}"
                )
                
                # Check if we should take action
                if (abs(decision["position"]) >= self.position_threshold and
                    decision["confidence"] >= self.min_confidence):
                    
                    # Here you would implement actual trade execution
                    self.logger.info(
                        f"Would execute trade: "
                        f"{'LONG' if decision['position'] > 0 else 'SHORT'} "
                        f"with size {abs(decision['position']):.2f}"
                    )
                    self.current_position = decision["position"]
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in live trading loop: {str(e)}")
                await asyncio.sleep(self.update_interval)
                
    def run(self):
        """Run the bot (blocking)."""
        asyncio.run(self.run_live()) 