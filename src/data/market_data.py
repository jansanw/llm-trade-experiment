from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yfinance as yf
import ccxt
from abc import ABC, abstractmethod
import pytz
from pandas_market_calendars import get_calendar

class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""
    
    @abstractmethod
    async def fetch_multi_timeframe_data(
        self,
        symbol: str,
        end_time: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple timeframes.
        
        Args:
            symbol: Trading symbol
            end_time: End time for historical data (defaults to current time)
            
        Returns:
            Tuple of DataFrames (hourly, 15min, 5min, 1min)
            Each DataFrame has columns: [timestamp, open, high, low, close, volume]
        """
        pass

class YFinanceProvider(MarketDataProvider):
    """YFinance implementation for futures and stocks."""
    
    def __init__(self):
        self.symbols_map = {
            "MNQ": "MNQ=F",  # Micro E-mini Nasdaq-100 Futures
            "SPY": "SPY",    # S&P 500 ETF
            # Add more mappings as needed
        }
        self.nyse = get_calendar('NYSE')  # NYSE calendar for market hours

    async def fetch_multi_timeframe_data(
        self,
        symbol: str,
        end_time: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch data from YFinance."""
        # Get current time in US/Eastern timezone (market time)
        now = datetime.now(pytz.timezone('US/Eastern'))
        
        # If end_time is not provided or is in the future, use current time
        if end_time is None or end_time > now:
            end_time = now
        
        # Ensure end_time is timezone-aware
        if end_time.tzinfo is None:
            end_time = pytz.timezone('US/Eastern').localize(end_time)
            
        # Get the last market day
        schedule = self.nyse.schedule(
            start_date=end_time - timedelta(days=30),
            end_date=end_time
        )
        if len(schedule) == 0:
            raise ValueError("No market days found in the date range")
            
        last_market_day = schedule.index[-1].to_pydatetime()
        last_market_day = pytz.timezone('US/Eastern').localize(last_market_day)
            
        yf_symbol = self.symbols_map.get(symbol, symbol)
        ticker = yf.Ticker(yf_symbol)
        
        # Calculate start times for each timeframe
        hourly_start = last_market_day - timedelta(days=10)  # 10 days for hourly data
        min15_start = last_market_day - timedelta(days=5)   # 5 days for 15min data
        min5_start = last_market_day - timedelta(days=2)    # 2 days for 5min data
        min1_start = last_market_day - timedelta(days=1)    # 1 day for 1min data
        
        # Fetch data for each timeframe
        try:
            hourly_df = ticker.history(start=hourly_start, end=last_market_day, interval="1h")
            min15_df = ticker.history(start=min15_start, end=last_market_day, interval="15m")
            min5_df = ticker.history(start=min5_start, end=last_market_day, interval="5m")
            min1_df = ticker.history(start=min1_start, end=last_market_day, interval="1m")
            
            # Ensure we have data
            if len(hourly_df) == 0 or len(min15_df) == 0 or len(min5_df) == 0 or len(min1_df) == 0:
                raise ValueError(f"No data available for {symbol}")
            
            # Take the last 100 rows for each timeframe
            hourly_df = hourly_df.tail(100)
            min15_df = min15_df.tail(100)
            min5_df = min5_df.tail(100)
            min1_df = min1_df.tail(100)
            
            # Standardize column names
            for df in [hourly_df, min15_df, min5_df, min1_df]:
                df.reset_index(inplace=True)
                df.columns = [c.lower() for c in df.columns]
                df.rename(columns={"datetime": "timestamp"}, inplace=True)
                
                # Ensure timestamps are timezone-aware
                if df["timestamp"].dt.tz is None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize('US/Eastern')
            
            return hourly_df, min15_df, min5_df, min1_df
            
        except Exception as e:
            raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

class CryptoProvider(MarketDataProvider):
    """Cryptocurrency data provider using CCXT."""
    
    def __init__(self, exchange_id: str = "binance"):
        self.exchange = getattr(ccxt, exchange_id)()
        
    async def fetch_multi_timeframe_data(
        self,
        symbol: str,
        end_time: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch data from crypto exchange."""
        # Get current time in UTC
        now = datetime.now(pytz.UTC)
        
        # If end_time is not provided or is in the future, use current time
        if end_time is None or end_time > now:
            end_time = now
            
        # Ensure end_time is timezone-aware
        if end_time.tzinfo is None:
            end_time = pytz.UTC.localize(end_time)
            
        timeframes = {
            "1h": 100 * 3600 * 1000,  # 100 hours in milliseconds
            "15m": 100 * 15 * 60 * 1000,
            "5m": 100 * 5 * 60 * 1000,
            "1m": 100 * 60 * 1000
        }
        
        dfs = []
        for timeframe in timeframes.keys():
            since = int(end_time.timestamp() * 1000) - timeframes[timeframe]
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since)
                
                if not ohlcv:
                    raise ValueError(f"No data available for {symbol} at {timeframe} timeframe")
                    
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                dfs.append(df)
            except Exception as e:
                raise ValueError(f"Error fetching {timeframe} data for {symbol}: {str(e)}")
            
        return tuple(dfs)

class MarketDataFetcher:
    """Factory class for creating appropriate market data provider."""
    
    def __init__(self):
        self.providers = {
            "futures": YFinanceProvider(),
            "stocks": YFinanceProvider(),
            "crypto": CryptoProvider()
        }
        
    def get_provider(self, asset_type: str) -> MarketDataProvider:
        """Get the appropriate provider for the asset type."""
        return self.providers[asset_type.lower()]
        
    @staticmethod
    def detect_asset_type(symbol: str) -> str:
        """Detect asset type from symbol."""
        if symbol in ["MNQ", "ES", "NQ"]:
            return "futures"
        elif symbol.endswith("USDT") or symbol.endswith("BTC"):
            return "crypto"
        else:
            return "stocks" 