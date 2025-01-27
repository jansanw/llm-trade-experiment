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
    """Prompt generator focused on fair value gaps and ICT concepts."""
    
    def _find_swing_points(self, df: pd.DataFrame, window: int = 10) -> tuple:
        """Find prominent swing highs and lows in the data.
        
        Args:
            df: DataFrame with OHLCV data
            window: Number of bars to look back/forward for swing point confirmation
            
        Returns:
            tuple: (swing_highs, swing_lows) DataFrames with timestamp and price
        """
        # Parameters for prominence
        min_price_distance_pct = 0.005  # 0.5% minimum price movement
        min_time_distance = pd.Timedelta(minutes=30)  # Minimum time between swings
        
        # Ensure index is datetime
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        highs = []
        lows = []
        last_high_time = None
        last_low_time = None
        last_high_price = None
        last_low_price = None
        
        for i in range(window, len(df) - window):
            current_price = df.iloc[i]['high']
            current_time = df.index[i]
            
            # Check if this is a swing high
            if all(df.iloc[i]['high'] > df.iloc[i-j]['high'] for j in range(1, window+1)) and \
               all(df.iloc[i]['high'] > df.iloc[i+j]['high'] for j in range(1, window+1)):
                
                # Check price distance from last high
                if last_high_price is None or \
                   abs(current_price - last_high_price) / last_high_price > min_price_distance_pct:
                    
                    # Check time distance from last high
                    if last_high_time is None or \
                       (current_time - last_high_time) > min_time_distance:
                        
                        highs.append({
                            'timestamp': current_time,
                            'price': current_price
                        })
                        last_high_time = current_time
                        last_high_price = current_price
            
            current_price = df.iloc[i]['low']
            # Check if this is a swing low
            if all(df.iloc[i]['low'] < df.iloc[i-j]['low'] for j in range(1, window+1)) and \
               all(df.iloc[i]['low'] < df.iloc[i+j]['low'] for j in range(1, window+1)):
                
                # Check price distance from last low
                if last_low_price is None or \
                   abs(current_price - last_low_price) / last_low_price > min_price_distance_pct:
                    
                    # Check time distance from last low
                    if last_low_time is None or \
                       (current_time - last_low_time) > min_time_distance:
                        
                        lows.append({
                            'timestamp': current_time,
                            'price': current_price
                        })
                        last_low_time = current_time
                        last_low_price = current_price
        
        # Convert to DataFrames
        highs_df = pd.DataFrame(highs)
        lows_df = pd.DataFrame(lows)
        
        # Further filter to keep only the most prominent points
        if not highs_df.empty:
            # Sort by price and keep top 5 highest highs
            highs_df = highs_df.nlargest(5, 'price')
        
        if not lows_df.empty:
            # Sort by price and keep top 5 lowest lows
            lows_df = lows_df.nsmallest(5, 'price')
        
        return (highs_df, lows_df)
        
    def _is_fvg_invalidated(self, gap: dict, df: pd.DataFrame) -> bool:
        """Check if a fair value gap has been invalidated.
        
        A gap is considered invalidated if price has traded through it twice.
        """
        # Get data after the gap
        gap_time = pd.to_datetime(gap['timestamp'])
        later_data = df[df['timestamp'] > gap_time]
        
        if later_data.empty:
            return False
            
        # Count how many times price traded through the gap
        invalidations = 0
        in_gap = False
        
        for _, bar in later_data.iterrows():
            # Price entering gap
            if not in_gap and ((bar['low'] <= gap['gap_high'] and bar['high'] >= gap['gap_low'])):
                in_gap = True
                invalidations += 1
            # Price leaving gap
            elif in_gap and ((bar['high'] < gap['gap_low'] or bar['low'] > gap['gap_high'])):
                in_gap = False
                
            if invalidations >= 2:
                return True
                
        return False
        
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
                gap = {
                    'timestamp': c2['timestamp'],
                    'type': 'bullish',
                    'gap_low': c3['high'],
                    'gap_high': c1['low']
                }
                # Only add if not invalidated
                if not self._is_fvg_invalidated(gap, df):
                    gaps.append(gap)
                    
            elif not bullish and c1['high'] < c3['low']:
                gap = {
                    'timestamp': c2['timestamp'],
                    'type': 'bearish',
                    'gap_low': c1['high'],
                    'gap_high': c3['low']
                }
                # Only add if not invalidated
                if not self._is_fvg_invalidated(gap, df):
                    gaps.append(gap)
                    
        # Convert to DataFrame
        gaps_df = pd.DataFrame(gaps)
            
        return gaps_df
        
    def _find_unified_swings(self, hourly_df: pd.DataFrame, min15_df: pd.DataFrame, 
                            min5_df: pd.DataFrame, min1_df: pd.DataFrame) -> tuple:
        """Find a unified series of swing points across all timeframes.
        
        Newer swings at similar price levels occlude older ones.
        Returns swings ordered by price (highest to lowest for highs, lowest to highest for lows).
        """
        # Parameters for prominence
        min_price_distance_pct = 0.005  # 0.5% minimum price movement
        occlusion_price_pct = 0.001     # 0.1% price difference to consider points at same level
        
        # Collect all potential swing points
        all_highs = []
        all_lows = []
        
        # Process each timeframe with appropriate window sizes
        for df, window in [
            (hourly_df, 3),   # 3-bar confirmation for hourly
            (min15_df, 4),    # 4-bar confirmation for 15min
            (min5_df, 6),     # 6-bar confirmation for 5min
            (min1_df, 10)     # 10-bar confirmation for 1min
        ]:
            if df.empty:
                continue
                
            for i in range(window, len(df) - window):
                # Check for swing high
                if all(df.iloc[i]['high'] > df.iloc[i-j]['high'] for j in range(1, window+1)) and \
                   all(df.iloc[i]['high'] > df.iloc[i+j]['high'] for j in range(1, window+1)):
                    all_highs.append({
                        'timestamp': df.iloc[i]['timestamp'],
                        'price': df.iloc[i]['high']
                    })
                
                # Check for swing low
                if all(df.iloc[i]['low'] < df.iloc[i-j]['low'] for j in range(1, window+1)) and \
                   all(df.iloc[i]['low'] < df.iloc[i+j]['low'] for j in range(1, window+1)):
                    all_lows.append({
                        'timestamp': df.iloc[i]['timestamp'],
                        'price': df.iloc[i]['low']
                    })
        
        # Convert to DataFrames
        highs_df = pd.DataFrame(all_highs)
        lows_df = pd.DataFrame(all_lows)
        
        if highs_df.empty or lows_df.empty:
            return (pd.DataFrame(), pd.DataFrame())
        
        # Sort by timestamp to process newer points first
        highs_df = highs_df.sort_values('timestamp', ascending=False)
        lows_df = lows_df.sort_values('timestamp', ascending=False)
        
        # Filter out occluded swing points
        filtered_highs = []
        filtered_lows = []
        
        # Process highs - newer points occlude older ones at similar prices
        for _, high in highs_df.iterrows():
            # Check if this point is near any existing filtered point
            is_occluded = any(
                abs(high['price'] - existing['price']) / existing['price'] < occlusion_price_pct
                for existing in filtered_highs
            )
            if not is_occluded:
                filtered_highs.append(high)
        
        # Process lows - newer points occlude older ones at similar prices
        for _, low in lows_df.iterrows():
            # Check if this point is near any existing filtered point
            is_occluded = any(
                abs(low['price'] - existing['price']) / existing['price'] < occlusion_price_pct
                for existing in filtered_lows
            )
            if not is_occluded:
                filtered_lows.append(low)
        
        # Convert back to DataFrames
        highs_df = pd.DataFrame(filtered_highs)
        lows_df = pd.DataFrame(filtered_lows)
        
        # Sort by price
        if not highs_df.empty:
            highs_df = highs_df.sort_values('price', ascending=False)
        if not lows_df.empty:
            lows_df = lows_df.sort_values('price', ascending=True)
        
        return (highs_df, lows_df)
        
    def generate(self, hourly_df: pd.DataFrame, min15_df: pd.DataFrame, 
                min5_df: pd.DataFrame, min1_df: pd.DataFrame, 
                additional_context: Optional[Dict] = None) -> str:
        current_price = min1_df.iloc[-1]['close']
        current_time = min1_df.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M')
        
        prompt = "You are a professional futures trader specializing in ICT concepts and market structure analysis.\n\n"
        prompt += f"Current Time: {current_time}\n"
        prompt += f"Current Price: {current_price:.2f}\n\n"
        
        # Add market structure analysis
        prompt += "Market Structure Analysis:\n"
        
        # Add swing points with context
        swing_highs, swing_lows = self._find_unified_swings(hourly_df, min15_df, min5_df, min1_df)
        
        if not swing_highs.empty:
            prompt += "\nKey Resistance Levels (Swing Highs):\n"
            for _, high in swing_highs.iterrows():
                timestamp_str = high['timestamp'].strftime('%Y-%m-%d %H:%M')
                distance = ((high['price'] - current_price) / current_price) * 100
                prompt += f"- {high['price']:.2f} ({distance:+.2f}% from current price) formed at {timestamp_str}\n"
                
        if not swing_lows.empty:
            prompt += "\nKey Support Levels (Swing Lows):\n"
            for _, low in swing_lows.iterrows():
                timestamp_str = low['timestamp'].strftime('%Y-%m-%d %H:%M')
                distance = ((low['price'] - current_price) / current_price) * 100
                prompt += f"- {low['price']:.2f} ({distance:+.2f}% from current price) formed at {timestamp_str}\n"
        
        # Add trend analysis
        prompt += "\nTrend Analysis:\n"
        for name, df in [("Hourly", hourly_df), ("15min", min15_df), ("5min", min5_df)]:
            if df.empty:
                continue
            recent = df.tail(10)
            price_change = ((recent.iloc[-1]['close'] - recent.iloc[0]['close']) / recent.iloc[0]['close']) * 100
            ema20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema50 = df['close'].ewm(span=50).mean().iloc[-1]
            
            trend = "Bullish" if ema20 > ema50 else "Bearish"
            prompt += f"\n{name} Timeframe:\n"
            prompt += f"- Price change last 10 periods: {price_change:+.2f}%\n"
            prompt += f"- EMA trend: {trend} (EMA20: {ema20:.2f} vs EMA50: {ema50:.2f})\n"
            
        # Add volume analysis
        prompt += "\nVolume Analysis:\n"
        for name, df in [("Hourly", hourly_df), ("15min", min15_df)]:
            if df.empty:
                continue
            recent_vol = df['volume'].tail(10)
            avg_vol = recent_vol.mean()
            last_vol = recent_vol.iloc[-1]
            vol_change = ((last_vol - avg_vol) / avg_vol) * 100
            prompt += f"\n{name} Volume:\n"
            prompt += f"- Current vs 10-period average: {vol_change:+.2f}%\n"
            
        # Add active FVGs
        prompt += "\nFair Value Gaps (Active & Non-invalidated):\n"
        for name, df in [("Hourly", hourly_df), ("15min", min15_df), ("5min", min5_df)]:
            bullish_gaps = self._find_fvg(df, bullish=True)
            bearish_gaps = self._find_fvg(df, bullish=False)
            
            if not bullish_gaps.empty or not bearish_gaps.empty:
                prompt += f"\n{name} Timeframe FVGs:\n"
                
                if not bullish_gaps.empty:
                    prompt += "\nBullish Gaps (Potential Support):\n"
                    for _, gap in bullish_gaps.iterrows():
                        timestamp_str = gap['timestamp'].strftime('%Y-%m-%d %H:%M')
                        distance = ((gap['gap_low'] - current_price) / current_price) * 100
                        prompt += f"- {gap['gap_low']:.2f} to {gap['gap_high']:.2f} ({distance:+.2f}% from current price) formed at {timestamp_str}\n"
                
                if not bearish_gaps.empty:
                    prompt += "\nBearish Gaps (Potential Resistance):\n"
                    for _, gap in bearish_gaps.iterrows():
                        timestamp_str = gap['timestamp'].strftime('%Y-%m-%d %H:%M')
                        distance = ((gap['gap_low'] - current_price) / current_price) * 100
                        prompt += f"- {gap['gap_low']:.2f} to {gap['gap_high']:.2f} ({distance:+.2f}% from current price) formed at {timestamp_str}\n"
        
        if additional_context:
            prompt += f"\nAdditional Context:\n{additional_context}\n"
        
        prompt += "\nBased on the above market structure analysis, provide a detailed trading plan:\n"
        prompt += "1. Position (-1.0 for full short to 1.0 for full long)\n"
        prompt += "2. Confidence level (0.0 to 1.0)\n"
        prompt += "3. Take-profit price - use nearest significant resistance for longs or support for shorts\n"
        prompt += "4. Stop-loss price - use market structure (swings/FVGs) to define invalidation level\n"
        prompt += "5. Detailed reasoning including:\n"
        prompt += "   - Primary market structure levels being used\n"
        prompt += "   - How the multi-timeframe trend aligns with the trade\n"
        prompt += "   - Volume confirmation/concerns\n"
        prompt += "   - Risk:reward ratio justification\n\n"
        prompt += "Format your response as a JSON object with keys: position, confidence, take_profit, stop_loss, reasoning\n"
        prompt += "Note: take_profit and stop_loss should be absolute price levels based on the current price of " + f"{current_price:.2f}"
        
        return prompt 