from datetime import datetime, timedelta
import numpy as np
import random
import pandas as pd
import logging
from typing import Dict, Optional
from .base import LLMProvider

class MockProvider(LLMProvider):
    """Mock LLM provider that simulates different trading behaviors for testing."""
    
    def __init__(self, behavior: str = 'trend_follower'):
        """Initialize mock provider.
        
        Args:
            behavior: Trading behavior to simulate. One of:
                - trend_follower: Takes positions in direction of recent trend
                - mean_reverter: Takes positions against recent trend 
                - random: Takes random positions
        """
        self.behavior = behavior
        self.logger = logging.getLogger(__name__)
        
    def _calculate_trend(self, df: pd.DataFrame) -> float:
        """Calculate trend score from -1 to 1 based on price action."""
        if df.empty:
            return 0.0
            
        df = df.copy()
        df["sma5"] = df["close"].rolling(5).mean()
        df["sma20"] = df["close"].rolling(20).mean()
        df["roc"] = df["close"].pct_change(5)
        
        last_bar = df.iloc[-1]
        
        # Trend signals
        sma_trend = 1 if last_bar["sma5"] > last_bar["sma20"] else -1
        roc_trend = 1 if last_bar["roc"] > 0 else -1
        
        # Combine signals
        trend = (sma_trend + roc_trend) / 2
        return trend
        
    async def get_trading_decision(
        self,
        hourly_df: pd.DataFrame,
        min15_df: pd.DataFrame,
        min5_df: pd.DataFrame,
        min1_df: pd.DataFrame,
        additional_context: Optional[Dict] = None
    ) -> Dict:
        """Get trading decision based on market data."""
        self.logger.info("Starting trend calculation...")
        
        # Calculate trends for each timeframe
        h1_trend = self._calculate_trend(hourly_df)
        m15_trend = self._calculate_trend(min15_df)
        m5_trend = self._calculate_trend(min5_df)
        m1_trend = self._calculate_trend(min1_df)
        
        self.logger.debug(f"Hourly trend: {h1_trend:.2f}")
        self.logger.debug(f"15min trend: {m15_trend:.2f}")
        self.logger.debug(f"5min trend: {m5_trend:.2f}")
        self.logger.debug(f"1min trend: {m1_trend:.2f}")
        
        # Weight the trends (higher weight on longer timeframes)
        trend = (0.4 * h1_trend + 0.3 * m15_trend + 0.2 * m5_trend + 0.1 * m1_trend)
        self.logger.info(f"Weighted trend: {trend:.2f}")
        
        # Get current price
        current_price = min1_df.iloc[-1]["close"]
        self.logger.debug(f"Current price: {current_price}")
        
        # Generate decision based on behavior
        self.logger.info(f"Using {self.behavior} strategy...")
        
        if self.behavior == "trend_follower":
            position = np.sign(trend)
            confidence = abs(trend)
        elif self.behavior == "mean_reverter":
            position = -np.sign(trend)
            confidence = abs(trend)
        else:  # random
            position = np.random.choice([-1, 0, 1])
            confidence = np.random.random()
            
        # Set take-profit and stop-loss levels
        if abs(position) > 0:
            take_profit = current_price * (1 + 0.01 * position)  # 1% target
            stop_loss = current_price * (1 - 0.005 * position)  # 0.5% stop
        else:
            take_profit = None
            stop_loss = None
            
        decision = {
            "position": position,
            "confidence": confidence,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "reasoning": f"Taking {'long' if position > 0 else 'short' if position < 0 else 'neutral'} position based on {'strong' if confidence > 0.7 else 'moderate' if confidence > 0.3 else 'weak'} {'uptrend' if trend > 0 else 'downtrend'} detected across timeframes (H1: {h1_trend:.2f}, 15M: {m15_trend:.2f}, 5M: {m5_trend:.2f}, 1M: {m1_trend:.2f})"
        }
        
        self.logger.info(f"Generated decision: position={position}, confidence={confidence*100:.1f}%")
        return decision 