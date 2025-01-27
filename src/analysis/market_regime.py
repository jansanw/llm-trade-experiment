import pandas as pd
import numpy as np
from enum import Enum
from typing import Tuple, Dict

class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING_LOW_VOL = "ranging_low_vol"
    RANGING_HIGH_VOL = "ranging_high_vol"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"

class MarketRegimeDetector:
    """Detects market regime using multiple indicators and timeframes."""
    
    def __init__(self, 
                 trend_window: int = 20,
                 vol_window: int = 20,
                 breakout_std: float = 2.0,
                 trend_threshold: float = 0.6):
        """Initialize detector with parameters.
        
        Args:
            trend_window: Window for trend calculations
            vol_window: Window for volatility calculations
            breakout_std: Standard deviations for breakout detection
            trend_threshold: Threshold for trend strength (0.0 to 1.0)
        """
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.breakout_std = breakout_std
        self.trend_threshold = trend_threshold
        
    def _calculate_trend_strength(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate trend strength using multiple indicators.
        
        Returns:
            Tuple[float, float]: (trend_strength, trend_direction)
            trend_strength: 0.0 to 1.0 (stronger trend)
            trend_direction: -1.0 to 1.0 (down to up)
        """
        # Calculate EMAs
        ema20 = df['close'].ewm(span=20).mean()
        ema50 = df['close'].ewm(span=50).mean()
        
        # ADX for trend strength
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()
        
        # Combine indicators for trend strength and direction
        trend_strength = min(1.0, adx.iloc[-1] / 100)
        
        # Trend direction from EMAs and DI
        ema_direction = 1 if ema20.iloc[-1] > ema50.iloc[-1] else -1
        di_direction = 1 if plus_di.iloc[-1] > minus_di.iloc[-1] else -1
        trend_direction = ema_direction if ema_direction == di_direction else 0
        
        return trend_strength, trend_direction
        
    def _detect_volatility_regime(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Detect if we're in a high volatility regime.
        
        Returns:
            Tuple[bool, float]: (is_high_vol, vol_percentile)
        """
        # Calculate historical volatility
        returns = np.log(df['close'] / df['close'].shift(1))
        rolling_std = returns.rolling(self.vol_window).std() * np.sqrt(252)
        
        # Get current volatility percentile
        vol_percentile = (rolling_std.iloc[-1] - rolling_std.min()) / (rolling_std.max() - rolling_std.min())
        is_high_vol = vol_percentile > 0.7
        
        return is_high_vol, vol_percentile
        
    def _detect_breakout(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Detect if we're in a breakout.
        
        Returns:
            Tuple[bool, float]: (is_breakout, breakout_strength)
        """
        # Calculate Bollinger Bands
        rolling_mean = df['close'].rolling(self.trend_window).mean()
        rolling_std = df['close'].rolling(self.trend_window).std()
        
        upper_band = rolling_mean + (rolling_std * self.breakout_std)
        lower_band = rolling_mean - (rolling_std * self.breakout_std)
        
        # Check if price is outside bands
        current_price = df['close'].iloc[-1]
        is_breakout = current_price > upper_band.iloc[-1] or current_price < lower_band.iloc[-1]
        
        # Calculate breakout strength
        if is_breakout:
            if current_price > upper_band.iloc[-1]:
                strength = (current_price - upper_band.iloc[-1]) / rolling_std.iloc[-1]
            else:
                strength = (lower_band.iloc[-1] - current_price) / rolling_std.iloc[-1]
        else:
            strength = 0.0
            
        return is_breakout, min(1.0, strength / 2)
        
    def detect_regime(self, hourly_df: pd.DataFrame, min15_df: pd.DataFrame) -> Dict:
        """Detect current market regime using multiple timeframes.
        
        Returns:
            Dict with regime info:
            - regime: MarketRegime enum
            - confidence: 0.0 to 1.0
            - details: Dict with supporting metrics
        """
        # Get trend info from both timeframes
        h_trend_str, h_trend_dir = self._calculate_trend_strength(hourly_df)
        m15_trend_str, m15_trend_dir = self._calculate_trend_strength(min15_df)
        
        # Get volatility info
        h_high_vol, h_vol_pct = self._detect_volatility_regime(hourly_df)
        m15_high_vol, m15_vol_pct = self._detect_volatility_regime(min15_df)
        
        # Get breakout info
        h_breakout, h_break_str = self._detect_breakout(hourly_df)
        m15_breakout, m15_break_str = self._detect_breakout(min15_df)
        
        # Combine metrics
        trend_strength = (h_trend_str * 0.7 + m15_trend_str * 0.3)
        trend_direction = h_trend_dir if abs(h_trend_dir) > 0 else m15_trend_dir
        is_high_vol = h_high_vol or m15_high_vol
        vol_percentile = max(h_vol_pct, m15_vol_pct)
        is_breakout = h_breakout or m15_breakout
        breakout_strength = max(h_break_str, m15_break_str)
        
        # Determine regime
        if is_breakout and breakout_strength > 0.5:
            regime = MarketRegime.BREAKOUT
            confidence = breakout_strength
        elif trend_strength > self.trend_threshold:
            if trend_direction > 0:
                regime = MarketRegime.TRENDING_UP
            else:
                regime = MarketRegime.TRENDING_DOWN
            confidence = trend_strength
        else:
            if is_high_vol:
                regime = MarketRegime.RANGING_HIGH_VOL
            else:
                regime = MarketRegime.RANGING_LOW_VOL
            confidence = 1 - trend_strength
            
        # Check for potential reversal
        if trend_strength > 0.4 and breakout_strength > 0.3 and trend_direction * h_trend_dir < 0:
            regime = MarketRegime.REVERSAL
            confidence = min(trend_strength, breakout_strength)
            
        return {
            "regime": regime,
            "confidence": confidence,
            "details": {
                "trend_strength": trend_strength,
                "trend_direction": trend_direction,
                "volatility_percentile": vol_percentile,
                "is_high_volatility": is_high_vol,
                "breakout_strength": breakout_strength
            }
        } 