from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import asyncio
import logging
from src.llm.base import LLMProvider
from src.data.market_data import MarketDataFetcher, MarketDataProvider

class TradingBot:
    """Main trading bot class that coordinates LLM decisions with market data."""
    
    def __init__(self, symbol: str, data_fetcher, llm, 
                 max_position_size: float = 1.0,
                 min_confidence: float = 0.6,
                 min_risk_reward: float = 1.5):
        """Initialize the trading bot.
        
        Args:
            symbol: Trading symbol (e.g. SPY, QQQ)
            data_fetcher: Market data fetcher instance
            llm: LLM provider instance
            max_position_size: Maximum position size (1.0 = 100%)
            min_confidence: Minimum confidence required to take a trade
            min_risk_reward: Minimum risk/reward ratio required
        """
        self.symbol = symbol
        self.data_fetcher = data_fetcher
        self.llm = llm
        self.max_position_size = max_position_size
        self.min_confidence = min_confidence
        self.min_risk_reward = min_risk_reward
        self.logger = logging.getLogger(__name__)
        
    def _calculate_position_size(self, decision: dict) -> float:
        """Calculate position size based on confidence and risk/reward.
        
        Args:
            decision: Trading decision dict with position, confidence, take_profit, stop_loss
            
        Returns:
            float: Position size (0.0 to max_position_size)
        """
        if abs(decision.get('position', 0)) < 0.1 or decision.get('confidence', 0) < self.min_confidence:
            return 0.0
            
        # Calculate risk/reward ratio
        current_price = float(decision.get('current_price', 0))
        take_profit = float(decision.get('take_profit', 0))
        stop_loss = float(decision.get('stop_loss', 0))
        
        if not all([current_price, take_profit, stop_loss]):
            return 0.0
            
        # Calculate potential profit and loss
        if decision['position'] > 0:  # Long
            potential_profit = take_profit - current_price
            potential_loss = current_price - stop_loss
        else:  # Short
            potential_profit = current_price - take_profit
            potential_loss = stop_loss - current_price
            
        if potential_loss <= 0:
            return 0.0
            
        risk_reward = potential_profit / potential_loss
        
        if risk_reward < self.min_risk_reward:
            self.logger.info(f"Risk/reward {risk_reward:.2f} below minimum {self.min_risk_reward}")
            return 0.0
            
        # Scale position size by confidence and risk/reward
        confidence_factor = min(1.0, decision.get('confidence', 0))
        risk_reward_factor = min(1.0, risk_reward / (2 * self.min_risk_reward))
        
        position_size = self.max_position_size * confidence_factor * risk_reward_factor
        return round(position_size, 2)
        
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
            
            if min1_df.empty:
                return {
                    "position": 0.0,
                    "confidence": 0.0,
                    "reasoning": "No market data available"
                }
                
            current_price = min1_df.iloc[-1]['close']
            
            # Get trading decision from LLM
            decision = await self.llm.get_trading_decision(
                hourly_df=hourly_df,
                min15_df=min15_df,
                min5_df=min5_df,
                min1_df=min1_df
            )
            
            # Add current price for position sizing
            decision['current_price'] = current_price
            
            # Calculate position size
            raw_position = decision.get('position', 0)
            position_size = self._calculate_position_size(decision)
            
            # Scale the position by calculated size
            decision['position'] = position_size if raw_position > 0 else -position_size
            
            # Log decision details
            self.logger.info(
                f"Decision: pos={decision['position']:.2f} (raw={raw_position:.2f}), "
                f"conf={decision.get('confidence', 0):.2f}, "
                f"tp={decision.get('take_profit', 0):.2f}, "
                f"sl={decision.get('stop_loss', 0):.2f}"
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