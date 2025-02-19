from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import asyncio
import logging
from src.llm.base import LLMProvider
from src.data.market_data import MarketDataFetcher, MarketDataProvider
from src.analysis.market_regime import MarketRegimeDetector, MarketRegime
from src.prompts.generators import PromptV0, PromptFVG, PromptRaw

class TradingBot:
    """Main trading bot class that coordinates LLM decisions with market data."""
    
    def __init__(self, symbol: str, data_fetcher, llm, 
                 max_position_size: float = 1.0,
                 min_confidence: float = 0.6,
                 min_risk_reward: float = 1.5,
                 prompt_type: str = 'fvg'):
        """Initialize the trading bot.
        
        Args:
            symbol: Trading symbol (e.g. SPY, QQQ)
            data_fetcher: Market data fetcher instance
            llm: LLM provider instance
            max_position_size: Maximum position size (1.0 = 100%)
            min_confidence: Minimum confidence required to take a trade
            min_risk_reward: Minimum risk/reward ratio required
            prompt_type: Type of prompt generator to use ('v0', 'fvg', or 'raw')
        """
        self.symbol = symbol
        self.data_fetcher = data_fetcher
        self.llm = llm
        self.max_position_size = max_position_size
        self.min_confidence = min_confidence
        self.min_risk_reward = min_risk_reward
        self.regime_detector = MarketRegimeDetector()
        self.logger = logging.getLogger(__name__)
        
        # Set prompt generator based on type
        if prompt_type == 'v0':
            self.llm.prompt_generator = PromptV0()
        elif prompt_type == 'fvg':
            self.llm.prompt_generator = PromptFVG()
        elif prompt_type == 'raw':
            self.llm.prompt_generator = PromptRaw()
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        
    def _adjust_for_regime(self, decision: dict, regime_info: dict) -> dict:
        """Adjust trading decision based on market regime.
        
        Args:
            decision: Original trading decision
            regime_info: Market regime information
            
        Returns:
            dict: Adjusted trading decision
        """
        regime = regime_info['regime']
        regime_conf = regime_info['confidence']
        details = regime_info['details']
        
        # Adjust confidence based on regime alignment
        position = decision.get('position', 0)
        confidence = decision.get('confidence', 0)
        
        # In ranging markets, reduce position size and tighten stops
        if regime in [MarketRegime.RANGING_LOW_VOL, MarketRegime.RANGING_HIGH_VOL]:
            decision['position'] = position * 0.7  # Reduce position size
            
            # Tighten stops in high volatility
            if regime == MarketRegime.RANGING_HIGH_VOL:
                current_price = decision.get('current_price', 0)
                if position > 0:  # Long
                    new_sl = max(
                        decision.get('stop_loss', 0),
                        current_price - (current_price - decision.get('stop_loss', 0)) * 0.7
                    )
                else:  # Short
                    new_sl = min(
                        decision.get('stop_loss', 0),
                        current_price + (decision.get('stop_loss', 0) - current_price) * 0.7
                    )
                decision['stop_loss'] = new_sl
                
        # In trending markets, align with trend and potentially increase position
        elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            trend_alignment = (
                (regime == MarketRegime.TRENDING_UP and position > 0) or
                (regime == MarketRegime.TRENDING_DOWN and position < 0)
            )
            if trend_alignment:
                decision['confidence'] = min(1.0, confidence * (1 + regime_conf * 0.3))
            else:
                decision['confidence'] = confidence * 0.5
                
        # In breakout regimes, increase size if aligned with breakout
        elif regime == MarketRegime.BREAKOUT:
            breakout_alignment = (
                (details['trend_direction'] > 0 and position > 0) or
                (details['trend_direction'] < 0 and position < 0)
            )
            if breakout_alignment:
                decision['confidence'] = min(1.0, confidence * (1 + regime_conf * 0.5))
                
        # In reversal regimes, increase confidence if trading against old trend
        elif regime == MarketRegime.REVERSAL:
            reversal_alignment = (
                (details['trend_direction'] < 0 and position > 0) or
                (details['trend_direction'] > 0 and position < 0)
            )
            if reversal_alignment:
                decision['confidence'] = min(1.0, confidence * (1 + regime_conf * 0.4))
                
        # Add regime info to reasoning
        decision['reasoning'] = f"Market Regime: {regime.value} (conf: {regime_conf:.2f})\n" + decision.get('reasoning', '')
        
        return decision
        
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
            
            # Detect market regime
            regime_info = self.regime_detector.detect_regime(hourly_df, min15_df)
            
            # Get trading decision from LLM
            decision = await self.llm.get_trading_decision(
                hourly_df=hourly_df,
                min15_df=min15_df,
                min5_df=min5_df,
                min1_df=min1_df,
                additional_context={"market_regime": regime_info}
            )
            
            # Add current price for position sizing
            decision['current_price'] = current_price
            
            # Adjust decision based on market regime
            decision = self._adjust_for_regime(decision, regime_info)
            
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
                f"sl={decision.get('stop_loss', 0):.2f}, "
                f"regime={regime_info['regime'].value}"
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