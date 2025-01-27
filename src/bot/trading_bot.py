from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import asyncio
import logging
from src.llm.base import LLMProvider
from src.data.market_data import MarketDataFetcher, MarketDataProvider

class TradingBot:
    """Main trading bot class that coordinates LLM decisions with market data."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        symbol: str,
        update_interval: int = 60,  # seconds
        min_confidence: float = 0.7,
        position_threshold: float = 0.3
    ):
        """
        Initialize the trading bot.
        
        Args:
            llm_provider: LLM provider instance
            symbol: Trading symbol
            update_interval: How often to check for new signals (seconds)
            min_confidence: Minimum confidence required to take a position
            position_threshold: Minimum absolute position value to consider taking action
        """
        self.llm = llm_provider
        self.symbol = symbol
        self.update_interval = update_interval
        self.min_confidence = min_confidence
        self.position_threshold = position_threshold
        
        self.market_data = MarketDataFetcher()
        self.provider = self.market_data.get_provider(
            MarketDataFetcher.detect_asset_type(symbol)
        )
        
        self.current_position = 0.0
        self.last_decision_time = None
        self.last_decision = None
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    async def get_trading_decision(
        self,
        end_time: Optional[datetime] = None
    ) -> Dict:
        """
        Get trading decision from LLM based on current market data.
        
        Args:
            end_time: Optional end time for historical data
            
        Returns:
            Dictionary containing position, confidence, and reasoning
        """
        try:
            # Fetch market data
            hourly_df, min15_df, min5_df, min1_df = (
                await self.provider.fetch_multi_timeframe_data(
                    self.symbol,
                    end_time
                )
            )
            
            # Get additional context about current position
            context = {
                "current_position": self.current_position,
                "last_decision_time": self.last_decision_time,
                "last_decision": self.last_decision
            }
            
            # Get LLM decision
            decision = await self.llm.get_trading_decision(
                hourly_df,
                min15_df,
                min5_df,
                min1_df,
                context
            )
            
            self.last_decision = decision
            self.last_decision_time = datetime.now()
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error getting trading decision: {str(e)}")
            return {
                "position": 0.0,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }
            
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