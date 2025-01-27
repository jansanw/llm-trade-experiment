import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Optional

class BasePromptGenerator(ABC):
    """Base class for prompt generators."""
    
    @abstractmethod
    def generate(self, hourly_df: pd.DataFrame, min15_df: pd.DataFrame, 
                min5_df: pd.DataFrame, min1_df: pd.DataFrame, 
                additional_context: Optional[Dict] = None) -> str:
        """Generate a prompt from the market data."""
        pass

class PromptV0(BasePromptGenerator):
    """Original prompt generator with basic OHLCV analysis."""
    
    def generate(self, hourly_df: pd.DataFrame, min15_df: pd.DataFrame, 
                min5_df: pd.DataFrame, min1_df: pd.DataFrame, 
                additional_context: Optional[Dict] = None) -> str:
        # Get summary stats for each timeframe
        summaries = []
        for name, df in [
            ("hourly", hourly_df),
            ("15-minute", min15_df),
            ("5-minute", min5_df),
            ("1-minute", min1_df)
        ]:
            if df.empty:
                continue
                
            current = df.iloc[-1]
            recent = df.tail(10)
            
            price_change = ((current['close'] - recent.iloc[0]['close']) / recent.iloc[0]['close']) * 100
            volatility = recent['close'].std()
            volume_trend = "Increasing" if recent['volume'].is_monotonic_increasing else "Decreasing"
            
            summary = f"{name.capitalize()} candles:\n"
            summary += f"Current: Open={current['open']:.2f}, High={current['high']:.2f}, Low={current['low']:.2f}, Close={current['close']:.2f}\n"
            summary += "Recent trends:\n"
            summary += f"- Price change last 10 periods: {price_change:.2f}%\n"
            summary += f"- Volatility (10-period std): {volatility:.2f}\n"
            summary += f"- Volume trend: {volume_trend}\n"
            summaries.append(summary)
            
        current_price = min1_df.iloc[-1]['close']
        prompt = "You are a professional futures trader. Analyze the following market data and provide a trading decision.\n\n"
        prompt += f"Current Price: {current_price:.2f}\n\n"
        prompt += "Market Data:\n"
        prompt += "\n".join(f"{i+1}. Last 100 {summary}" for i, summary in enumerate(summaries))
        
        if additional_context:
            prompt += f"\nAdditional Context:\n{additional_context}\n"
            
        prompt += "\nBased on this data, provide a complete trading plan including:\n"
        prompt += "1. Position (-1.0 for full short to 1.0 for full long)\n"
        prompt += "2. Confidence level (0.0 to 1.0)\n"
        prompt += "3. Take-profit price level\n"
        prompt += "4. Stop-loss price level\n"
        prompt += "5. Brief explanation of your reasoning\n\n"
        prompt += "Format your response as a JSON object with keys: position, confidence, take_profit, stop_loss, reasoning\n"
        prompt += "Note: take_profit and stop_loss should be absolute price levels based on the current price of " + f"{current_price:.2f}"
        
        return prompt

class PromptFVG(BasePromptGenerator):
    """Prompt generator focused on fair value gaps."""
    
    def _find_fvg(self, df: pd.DataFrame, bullish: bool = True) -> pd.DataFrame:
        """Find fair value gaps in the data.
        
        A bullish FVG occurs when:
        - Low of candle 1 > High of candle 3
        - Candle 2 is the gap
        
        A bearish FVG occurs when:
        - High of candle 1 < Low of candle 3
        - Candle 2 is the gap
        """
        gaps = []
        for i in range(len(df) - 2):
            c1 = df.iloc[i]    # First candle
            c2 = df.iloc[i+1]  # Gap candle
            c3 = df.iloc[i+2]  # Third candle
            
            if bullish and c1['low'] > c3['high']:
                gaps.append({
                    'timestamp': c2.name,
                    'type': 'bullish',
                    'gap_low': c3['high'],
                    'gap_high': c1['low'],
                    'c1_open': c1['open'],
                    'c1_high': c1['high'],
                    'c1_low': c1['low'],
                    'c1_close': c1['close'],
                    'c3_open': c3['open'],
                    'c3_high': c3['high'],
                    'c3_low': c3['low'],
                    'c3_close': c3['close']
                })
            elif not bullish and c1['high'] < c3['low']:
                gaps.append({
                    'timestamp': c2.name,
                    'type': 'bearish',
                    'gap_low': c1['high'],
                    'gap_high': c3['low'],
                    'c1_open': c1['open'],
                    'c1_high': c1['high'],
                    'c1_low': c1['low'],
                    'c1_close': c1['close'],
                    'c3_open': c3['open'],
                    'c3_high': c3['high'],
                    'c3_low': c3['low'],
                    'c3_close': c3['close']
                })
        
        return pd.DataFrame(gaps) if gaps else pd.DataFrame()

    def generate(self, hourly_df: pd.DataFrame, min15_df: pd.DataFrame, 
                min5_df: pd.DataFrame, min1_df: pd.DataFrame, 
                additional_context: Optional[Dict] = None) -> str:
        current_price = min1_df.iloc[-1]['close']
        
        prompt = "You are a professional futures trader specializing in fair value gaps (FVG) analysis.\n\n"
        prompt += f"Current Price: {current_price:.2f}\n\n"
        prompt += "Below are all identified fair value gaps in different timeframes.\n"
        prompt += "For each gap, you have: timestamp, type (bullish/bearish), gap_low, gap_high, "
        prompt += "and OHLC data for the candles forming the gap (c1 and c3).\n\n"
        
        # Process each timeframe
        for name, df in [
            ("Hourly", hourly_df),
            ("15min", min15_df),
            ("5min", min5_df),
            ("1min", min1_df)
        ]:
            bullish_gaps = self._find_fvg(df, bullish=True)
            bearish_gaps = self._find_fvg(df, bullish=False)
            
            prompt += f"\n{name} Timeframe FVGs:\n"
            if bullish_gaps.empty and bearish_gaps.empty:
                prompt += "No fair value gaps found\n"
                continue
            
            if not bullish_gaps.empty:
                prompt += "\nBullish Gaps:\n"
                prompt += bullish_gaps.to_string(index=False) + "\n"
            
            if not bearish_gaps.empty:
                prompt += "\nBearish Gaps:\n"
                prompt += bearish_gaps.to_string(index=False) + "\n"
        
        if additional_context:
            prompt += f"\nAdditional Context:\n{additional_context}\n"
        
        prompt += "\nBased on the fair value gaps and current price, provide a complete trading plan:\n"
        prompt += "1. Position (-1.0 for full short to 1.0 for full long)\n"
        prompt += "2. Confidence level (0.0 to 1.0)\n"
        prompt += "3. Take-profit price level - consider using relevant FVG levels\n"
        prompt += "4. Stop-loss price level - consider using the opposite side of relevant FVGs\n"
        prompt += "5. Brief explanation of your reasoning, specifically mentioning:\n"
        prompt += "   - Which gaps are most relevant for entry\n"
        prompt += "   - Which gaps are being used for take-profit/stop-loss levels\n\n"
        prompt += "Format your response as a JSON object with keys: position, confidence, take_profit, stop_loss, reasoning\n"
        prompt += "Note: take_profit and stop_loss should be absolute price levels based on the current price of " + f"{current_price:.2f}"
        
        return prompt 