from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd

class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    async def get_trading_decision(
        self,
        hourly_candles: pd.DataFrame,
        candles_15m: pd.DataFrame,
        candles_5m: pd.DataFrame,
        candles_1m: pd.DataFrame,
        additional_context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Get trading decision from the LLM.
        
        Args:
            hourly_candles: DataFrame with 100 hourly OHLC candles
            candles_15m: DataFrame with 100 15-minute OHLC candles
            candles_5m: DataFrame with 100 5-minute OHLC candles
            candles_1m: DataFrame with 100 1-minute OHLC candles
            additional_context: Optional additional context for the LLM
            
        Returns:
            Dict containing:
                - 'position': float between -1.0 (full short) and 1.0 (full long)
                - 'confidence': float between 0.0 and 1.0
                - 'reasoning': str explaining the decision
        """
        pass

    def _format_prompt(
        self,
        hourly_candles: pd.DataFrame,
        candles_15m: pd.DataFrame,
        candles_5m: pd.DataFrame,
        candles_1m: pd.DataFrame,
        additional_context: Optional[Dict] = None
    ) -> str:
        """Format the prompt for the LLM."""
        prompt = """You are a professional futures trader. Analyze the following market data and provide a trading decision.

Market Data:
1. Last 100 hourly candles:
{hourly_summary}

2. Last 100 15-minute candles:
{candles_15m_summary}

3. Last 100 5-minute candles:
{candles_5m_summary}

4. Last 100 1-minute candles:
{candles_1m_summary}

Additional Context:
{additional_context}

Based on this data, should we go long or short? Provide:
1. Position (-1.0 for full short to 1.0 for full long)
2. Confidence level (0.0 to 1.0)
3. Brief explanation of your reasoning

Format your response as a JSON object with keys: position, confidence, reasoning"""

        # Create summaries of the data
        hourly_summary = self._create_data_summary(hourly_candles)
        candles_15m_summary = self._create_data_summary(candles_15m)
        candles_5m_summary = self._create_data_summary(candles_5m)
        candles_1m_summary = self._create_data_summary(candles_1m)
        
        context_str = str(additional_context) if additional_context else "None"
        
        return prompt.format(
            hourly_summary=hourly_summary,
            candles_15m_summary=candles_15m_summary,
            candles_5m_summary=candles_5m_summary,
            candles_1m_summary=candles_1m_summary,
            additional_context=context_str
        )

    def _create_data_summary(self, df: pd.DataFrame) -> str:
        """Create a summary of the OHLC data."""
        latest = df.iloc[-1]
        summary = f"""Current: Open={latest['open']:.2f}, High={latest['high']:.2f}, Low={latest['low']:.2f}, Close={latest['close']:.2f}
Recent trends:
- Price change last 10 periods: {((df['close'].iloc[-1] / df['close'].iloc[-10]) - 1) * 100:.2f}%
- Volatility (10-period std): {df['close'].rolling(10).std().iloc[-1]:.2f}
- Volume trend: {'Increasing' if df['volume'].iloc[-5:].mean() > df['volume'].iloc[-10:-5].mean() else 'Decreasing'}"""
        return summary 