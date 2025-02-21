from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import yfinance as yf
import ccxt
from abc import ABC, abstractmethod
import pytz
from pandas_market_calendars import get_calendar
import logging
import numpy as np
import asyncio

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
            "QQQ": "QQQ",    # Nasdaq-100 ETF
            "IWM": "IWM",    # Russell 2000 ETF
            "DIA": "DIA",    # Dow Jones ETF
            # Add more mappings as needed
        }
        self.nyse = get_calendar('NYSE')  # NYSE calendar for market hours
        self.logger = logging.getLogger(__name__)
        self.timeout = 10  # 10 second timeout

    async def get_candles(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get OHLCV candles for symbol by fetching 1m data and resampling."""
        try:
            self.logger.info(f"Fetching {interval} data for {symbol} from {start_time} to {end_time}")
            
            # Map symbol if needed
            yf_symbol = self.symbols_map.get(symbol, symbol)
            self.logger.info(f"Using yfinance symbol: {yf_symbol}")
            
            ticker = yf.Ticker(yf_symbol)
            
            # Get 1-minute data with timeout
            try:
                df = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: ticker.history(start=start_time, end=end_time, interval="1m")
                    ),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout fetching 1m data for {symbol}")
                raise ValueError("Data fetch timed out")
            except Exception as e:
                self.logger.error(f"Error during yfinance API call: {str(e)}")
                raise ValueError(f"yfinance API error: {str(e)}")
                
            self.logger.debug(f"Got {len(df)} 1m bars")
            
            if len(df) == 0:
                self.logger.error(f"Empty dataframe returned for {symbol} ({yf_symbol})")
                raise ValueError(f"No data available for {symbol} ({yf_symbol})")
                
            # Resample to target interval if needed
            if interval != "1m":
                interval_map = {
                    "5m": "5T",
                    "15m": "15T",
                    "1h": "1H"
                }
                resample_rule = interval_map.get(interval)
                if resample_rule:
                    df = df.resample(resample_rule, closed='left', label='left').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }).dropna()
                
            # Standardize column names
            df.reset_index(inplace=True)
            df.columns = [c.lower() for c in df.columns]
            df.rename(columns={"datetime": "timestamp"}, inplace=True)
            
            # Ensure timestamps are timezone-aware
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize('US/Eastern')
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching {interval} data for {symbol}: {str(e)}")
            raise ValueError(f"Failed to fetch {interval} data for {symbol}: {str(e)}")

    async def fetch_multi_timeframe_data(
        self,
        symbol: str,
        end_time: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch data from YFinance."""
        self.logger.info(f"Fetching data for {symbol}")
        
        # Get current time and look back 14 days
        now = datetime.now(pytz.timezone('US/Eastern'))
        if end_time is None:
            end_time = now
        elif not end_time.tzinfo:
            end_time = pytz.timezone('US/Eastern').localize(end_time)
            
        # Map the symbol if needed
        yf_symbol = self.symbols_map.get(symbol, symbol)
        ticker = yf.Ticker(yf_symbol)
        
        try:
            # Fetch 1-minute data in 7-day chunks
            self.logger.info("Fetching 1-minute data in chunks...")
            chunks = []
            current_end = end_time
            
            for i in range(2):  # 2 chunks to cover 14 days
                current_start = current_end - timedelta(days=7)
                
                chunk = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: ticker.history(start=current_start, end=current_end, interval="1m")
                    ),
                    timeout=self.timeout
                )
                
                if not chunk.empty:
                    # Ensure index is timezone-aware
                    if chunk.index.tz is None:
                        chunk.index = chunk.index.tz_localize('US/Eastern')
                    chunks.append(chunk)
                    
                current_end = current_start
                
            if not chunks:
                raise ValueError(f"No data available for {symbol}")
                
            # Combine chunks and sort by time
            df_1m = pd.concat(chunks).sort_index()
            
            # Resample to other timeframes
            df_5m = df_1m.resample('5min', closed='left', label='left').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            df_15m = df_1m.resample('15min', closed='left', label='left').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            df_1h = df_1m.resample('1h', closed='left', label='left').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Take last 100 bars for each timeframe
            df_1m = df_1m.tail(100)
            df_5m = df_5m.tail(100)
            df_15m = df_15m.tail(100)
            df_1h = df_1h.tail(100)
            
            # Standardize column names and ensure timezone-aware timestamps
            for df in [df_1h, df_15m, df_5m, df_1m]:
                # Save the index before resetting
                df['timestamp'] = df.index
                df.reset_index(drop=True, inplace=True)
                df.columns = [c.lower() for c in df.columns]
                # Convert timestamps to timezone-aware datetime if needed
                if df["timestamp"].dt.tz is None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize('US/Eastern')
                
            return df_1h, df_15m, df_5m, df_1m
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")

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
    """Class for fetching market data."""
    
    def __init__(self, symbol: str):
        """Initialize the market data fetcher.
        
        Args:
            symbol: Trading symbol (e.g. "SPY", "BTC-USD")
        """
        self.symbol = symbol
        self.logger = logging.getLogger(__name__)
        self.et_tz = pytz.timezone('US/Eastern')
        self.provider = YFinanceProvider()
        
    async def get_candles(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: str = "1m"
    ) -> pd.DataFrame:
        """Get OHLCV candles for the symbol.
        
        Args:
            start_time: Start time for data fetch
            end_time: End time for data fetch
            interval: Candle interval (1m, 5m, 15m, 1h)
            
        Returns:
            DataFrame with OHLCV data
        """
        self.logger.info(f"Fetching {interval} candles for {self.symbol} from {start_time} to {end_time}")
        return await self.provider.get_candles(self.symbol, interval, start_time, end_time)
        
    async def fetch_multi_timeframe_data(
        self,
        end_time: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch data for multiple timeframes.
        
        Args:
            end_time: End time for data fetch (defaults to current time)
            
        Returns:
            Tuple of DataFrames (hourly, 15min, 5min, 1min)
        """
        self.logger.info(f"Fetching multi-timeframe data for {self.symbol} up to {end_time}")
        
        try:
            return await self.provider.fetch_multi_timeframe_data(self.symbol, end_time)
            
        except Exception as e:
            self.logger.error(f"Error fetching multi-timeframe data: {str(e)}")
            raise ValueError(f"No data available for {self.symbol}")
            
    @staticmethod
    def detect_asset_type(symbol: str) -> str:
        """Detect asset type from symbol."""
        if symbol in ["BTC", "ETH"]:
            return "crypto"
        return "stock"
        
    def get_provider(self, asset_type: str) -> Any:
        """Get data provider for asset type."""
        if asset_type == "crypto":
            return CryptoProvider()
        return YFinanceProvider() 