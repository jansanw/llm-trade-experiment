#!/usr/bin/env python3
"""
Backtest Log Analyzer

This script analyzes trading logs to evaluate performance of different prompt types,
specifically comparing the 'uniform' and 'fvg' prompt types.

It extracts trading decisions, take-profit and stop-loss levels from logs,
fetches market data for those dates from Polygon.io, and calculates
performance metrics like win rate.
"""

import os
import re
import json
import glob
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import asyncio
from polygon import RESTClient
from dotenv import load_dotenv
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load environment variables (for Polygon API key)
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LogAnalyzer:
    """Class for analyzing trading logs and calculating performance metrics."""
    
    def __init__(self, logs_dir='logs'):
        """Initialize the log analyzer.
        
        Args:
            logs_dir: Directory containing the log files
        """
        self.logs_dir = logs_dir
        self.log_files = self._get_log_files()
        self.polygon_api_key = os.getenv("POLYGON_API_KEY")
        
        if not self.polygon_api_key:
            raise ValueError("POLYGON_API_KEY not found in environment variables")
            
        self.polygon_client = RESTClient(self.polygon_api_key)
        self.trades = []
        
    def _get_log_files(self):
        """Get all log files in the logs directory."""
        return sorted(glob.glob(os.path.join(self.logs_dir, "trading_*.log")), reverse=True)
    
    def _parse_log_date(self, log_file):
        """Extract the date from the log filename.
        
        Args:
            log_file: Path to log file
            
        Returns:
            datetime: Date of the log
        """
        # Extract date from filename (format: trading_YYYYMMDD_HHMMSS.log)
        match = re.search(r'trading_(\d{8})_(\d{6})\.log', os.path.basename(log_file))
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            try:
                return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            except ValueError:
                logger.warning(f"Could not parse date from filename: {log_file}")
        return None
    
    def _extract_prompt_type(self, log_content):
        """Extract the prompt type from the log content.
        
        Args:
            log_content: Content of the log file
            
        Returns:
            str: Prompt type ('uniform', 'fvg', 'raw', 'v0', or 'unknown')
        """
        # Check for specific prompt indicators 
        if "Each timeframe contains exactly 60 candles" in log_content:
            return "uniform"
        elif "The following data shows price action across multiple timeframes" in log_content and "60 candles" in log_content:
            return "uniform"
        elif "Fair Value Gaps (FVG)" in log_content:
            return "fvg"
        elif "ICT concepts" in log_content:
            return "fvg"  # Most likely FVG if it mentions ICT concepts
        elif "professional futures trader" in log_content and "market structure" in log_content:
            return "fvg"
        elif "Raw Market Data:" in log_content:
            return "raw"
        elif "You are a professional trader. Analyze the following raw market data" in log_content:
            return "raw"
        elif "Statistical Summary" in log_content:
            return "v0"
        
        # Additional check for prompt type from command line arguments in logs
        prompt_type_match = re.search(r"--prompt-type (\w+)", log_content)
        if prompt_type_match:
            prompt_type = prompt_type_match.group(1)
            return prompt_type
            
        return "unknown"
    
    def _extract_symbol(self, log_content):
        """Extract the trading symbol from the log content.
        
        Args:
            log_content: Content of the log file
            
        Returns:
            str: Trading symbol or None
        """
        # Look for symbol in the log
        match = re.search(r"Fetching data for (\w+)", log_content)
        if match:
            return match.group(1)
            
        # Try from command line arguments
        match = re.search(r"--symbol (\w+)", log_content)
        if match:
            return match.group(1)
            
        return None
    
    def _extract_current_time_price(self, log_content):
        """Extract the current time and price from the log content.
        
        Args:
            log_content: Content of the log file
            
        Returns:
            tuple: (datetime, float) or (None, None)
        """
        # Look for current time in the log
        time_match = re.search(r"Current Time: (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", log_content)
        price_match = re.search(r"Current Price: (\d+\.\d+)", log_content)
        
        if time_match and price_match:
            time_str = time_match.group(1)
            price = float(price_match.group(1))
            try:
                current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                return current_time, price
            except ValueError:
                logger.warning(f"Could not parse current time: {time_str}")
        
        return None, None
    
    def _extract_trading_decision(self, log_content):
        """Extract the trading decision from the log content.
        
        Args:
            log_content: Content of the log file
            
        Returns:
            dict: Trading decision or None
        """
        # Try to find structured JSON response with position, take_profit, stop_loss
        
        # Look for the 'current_position' structure in JSON
        json_matches = re.findall(r'"current_position":\s*{[^}]*?"position":\s*(-?\d+\.\d+)[^}]*?"confidence":\s*(\d+\.\d+)[^}]*?"take_profit":\s*(\d+\.\d+)[^}]*?"stop_loss":\s*(\d+\.\d+)[^}]*?}', log_content)
        if json_matches:
            for match in json_matches:
                position, confidence, take_profit, stop_loss = [float(val) for val in match]
                # Skip if position is close to zero
                if abs(position) < 0.2:
                    continue
                
                # Return a structured decision dict
                return {
                    "current_position": {
                        "position": position,
                        "confidence": confidence,
                        "take_profit": take_profit,
                        "stop_loss": stop_loss
                    }
                }
        
        # For older JSON format, try to extract from Final decision log entry
        final_decision_match = re.search(r"Final decision: ({.*?})", log_content, re.DOTALL)
        if final_decision_match:
            try:
                # Convert single quotes to double quotes for proper JSON
                json_str = final_decision_match.group(1).replace("'", '"').replace('\n', ' ')
                # Parse the JSON
                decision = json.loads(json_str)
                if "current_position" in decision and "take_profit" in decision["current_position"]:
                    return decision
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"Could not parse JSON from Final decision: {e}")
        
        # Check for raw position info in logger output
        position_match = re.search(r"Decision: pos=(-?\d+\.\d+).*?conf=(\d+\.\d+).*?tp=(\d+\.\d+).*?sl=(\d+\.\d+)", log_content)
        if position_match:
            position = float(position_match.group(1))
            # Skip if position is close to zero
            if abs(position) < 0.2:
                return None
                
            return {
                "current_position": {
                    "position": position,
                    "confidence": float(position_match.group(2)),
                    "take_profit": float(position_match.group(3)),
                    "stop_loss": float(position_match.group(4))
                }
            }
        
        return None
    
    async def fetch_market_data(self, symbol, date, timespan="minute", multiplier=1):
        """Fetch market data for a specific date.
        
        Args:
            symbol: Trading symbol
            date: Date to fetch data for
            timespan: Timespan of data (minute, hour, day, etc.)
            multiplier: Multiplier for timespan
            
        Returns:
            pd.DataFrame: Market data
        """
        start_date = date.strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        try:
            logger.info(f"Fetching market data for {symbol} from {start_date} to {end_date}")
            aggs = self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=end_date,
                limit=50000
            )
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': datetime.fromtimestamp(agg.timestamp / 1000),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume
            } for agg in aggs])
            
            if not df.empty:
                logger.info(f"Fetched {len(df)} data points for {symbol}")
            else:
                logger.warning(f"No data available for {symbol} on {start_date}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return pd.DataFrame()
    
    def check_trade_outcome(self, trade, market_data):
        """Check if a trade hit take-profit or stop-loss.
        
        Args:
            trade: Trade information
            market_data: Market data for the day
            
        Returns:
            str: 'win', 'loss', or 'inconclusive'
        """
        if not market_data.empty and "current_position" in trade:
            position = trade["current_position"]["position"]
            take_profit = trade["current_position"]["take_profit"]
            stop_loss = trade["current_position"]["stop_loss"]
            entry_price = trade["entry_price"]
            
            # Filter data to include only timestamps after entry
            filtered_data = market_data[market_data["timestamp"] > trade["timestamp"]]
            
            if filtered_data.empty:
                logger.warning(f"No market data available after trade timestamp: {trade['timestamp']}")
                return "inconclusive"
                
            if position > 0:  # Long position
                # Check if price hit take profit (high >= tp)
                if any(filtered_data["high"] >= take_profit):
                    first_hit = filtered_data[filtered_data["high"] >= take_profit].iloc[0]
                    logger.info(f"Long take-profit hit at {first_hit['timestamp']} (High: {first_hit['high']} >= TP: {take_profit})")
                    return "win"
                # Check if price hit stop loss (low <= sl)
                elif any(filtered_data["low"] <= stop_loss):
                    first_hit = filtered_data[filtered_data["low"] <= stop_loss].iloc[0]
                    logger.info(f"Long stop-loss hit at {first_hit['timestamp']} (Low: {first_hit['low']} <= SL: {stop_loss})")
                    return "loss"
            elif position < 0:  # Short position
                # Check if price hit take profit (low <= tp)
                if any(filtered_data["low"] <= take_profit):
                    first_hit = filtered_data[filtered_data["low"] <= take_profit].iloc[0]
                    logger.info(f"Short take-profit hit at {first_hit['timestamp']} (Low: {first_hit['low']} <= TP: {take_profit})")
                    return "win"
                # Check if price hit stop loss (high >= sl)
                elif any(filtered_data["high"] >= stop_loss):
                    first_hit = filtered_data[filtered_data["high"] >= stop_loss].iloc[0]
                    logger.info(f"Short stop-loss hit at {first_hit['timestamp']} (High: {first_hit['high']} >= SL: {stop_loss})")
                    return "loss"
            
            # If we got here, neither TP nor SL was hit during the day
            logger.info(f"Trade neither hit take-profit ({take_profit}) nor stop-loss ({stop_loss}) during the day")
                
        return "inconclusive"
    
    def calculate_metrics(self, trades):
        """Calculate performance metrics for a set of trades.
        
        Args:
            trades: List of trades
            
        Returns:
            dict: Performance metrics
        """
        if not trades:
            return {
                "win_rate": 0,
                "loss_rate": 0,
                "inconclusive_rate": 0,
                "avg_profit_pct": 0,
                "avg_loss_pct": 0,
                "profit_factor": 0,
                "total_trades": 0
            }
        
        wins = [t for t in trades if t["outcome"] == "win"]
        losses = [t for t in trades if t["outcome"] == "loss"]
        inconclusive = [t for t in trades if t["outcome"] == "inconclusive"]
        
        total = len(trades)
        win_rate = len(wins) / total if total > 0 else 0
        loss_rate = len(losses) / total if total > 0 else 0
        inconclusive_rate = len(inconclusive) / total if total > 0 else 0
        
        # Calculate profit/loss percentages
        for trade in trades:
            if "current_position" in trade and trade["entry_price"] > 0:
                position = trade["current_position"]["position"]
                take_profit = trade["current_position"]["take_profit"]
                stop_loss = trade["current_position"]["stop_loss"]
                entry = trade["entry_price"]
                
                if position > 0:  # Long
                    trade["profit_pct"] = (take_profit - entry) / entry * 100
                    trade["loss_pct"] = (stop_loss - entry) / entry * 100
                else:  # Short
                    trade["profit_pct"] = (entry - take_profit) / entry * 100
                    trade["loss_pct"] = (entry - stop_loss) / entry * 100
        
        # Average profit/loss percentages
        avg_profit_pct = np.mean([t.get("profit_pct", 0) for t in wins]) if wins else 0
        avg_loss_pct = np.mean([t.get("loss_pct", 0) for t in losses]) if losses else 0
        
        # Profit factor (total profit / total loss)
        total_profit = sum([t.get("profit_pct", 0) for t in wins])
        total_loss = abs(sum([t.get("loss_pct", 0) for t in losses]))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        return {
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "inconclusive_rate": inconclusive_rate,
            "avg_profit_pct": avg_profit_pct,
            "avg_loss_pct": avg_loss_pct,
            "profit_factor": profit_factor,
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "inconclusive": len(inconclusive)
        }
    
    async def analyze_logs(self):
        """Analyze all log files and calculate performance metrics."""
        # Counter for stats
        stats = {
            "total_logs": 0,
            "logs_with_decisions": 0,
            "logs_by_prompt_type": {}
        }
        
        for log_file in self.log_files:
            try:
                stats["total_logs"] += 1
                log_date = self._parse_log_date(log_file)
                if not log_date:
                    logger.warning(f"Could not parse date from filename: {log_file}")
                    continue
                
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                
                # Extract information from log
                prompt_type = self._extract_prompt_type(log_content)
                
                # Update prompt type stats
                if prompt_type not in stats["logs_by_prompt_type"]:
                    stats["logs_by_prompt_type"][prompt_type] = 0
                stats["logs_by_prompt_type"][prompt_type] += 1
                
                symbol = self._extract_symbol(log_content)
                current_time, current_price = self._extract_current_time_price(log_content)
                decision = self._extract_trading_decision(log_content)
                
                # Log some debugging information
                if prompt_type != "unknown":
                    logger.debug(f"Log {log_file}: Prompt type: {prompt_type}, Symbol: {symbol}, " +
                               f"Has time/price: {current_time is not None and current_price is not None}, " +
                               f"Has decision: {decision is not None}")
                
                if not all([symbol, current_time, current_price, decision]):
                    # Skip logs that don't have all required information
                    continue
                
                stats["logs_with_decisions"] += 1
                
                # Check if decision has a significant position
                position = decision.get("current_position", {}).get("position", 0)
                if abs(position) < 0.2:  # Skip very small or neutral positions
                    continue
                
                logger.info(f"Found trade in {log_file} - Prompt: {prompt_type}, Symbol: {symbol}, " +
                          f"Position: {position}, Entry: {current_price}")
                
                # Create trade record
                trade = {
                    "log_file": log_file,
                    "log_date": log_date,
                    "prompt_type": prompt_type,
                    "symbol": symbol,
                    "timestamp": current_time,
                    "entry_price": current_price,
                    "current_position": decision.get("current_position", {}),
                    "daily_bias": decision.get("daily_bias", {})
                }
                
                # Fetch market data for the day
                market_data = await self.fetch_market_data(symbol, current_time.date())
                
                # Determine trade outcome
                trade["outcome"] = self.check_trade_outcome(trade, market_data)
                
                # Add to trades list
                self.trades.append(trade)
                
                logger.info(f"Analyzed log: {log_file}, Prompt: {prompt_type}, Outcome: {trade['outcome']}")
                
            except Exception as e:
                logger.error(f"Error analyzing log {log_file}: {str(e)}")
        
        # Log some statistics about the analysis
        logger.info(f"Analysis summary: Processed {stats['total_logs']} logs, found {stats['logs_with_decisions']} logs with trading decisions")
        logger.info(f"Logs by prompt type: {stats['logs_by_prompt_type']}")
        
        # Group trades by prompt type
        prompt_types = {}
        for trade in self.trades:
            prompt_type = trade["prompt_type"]
            if prompt_type not in prompt_types:
                prompt_types[prompt_type] = []
            prompt_types[prompt_type].append(trade)
        
        # Calculate metrics for each prompt type
        metrics = {}
        for prompt_type, trades in prompt_types.items():
            metrics[prompt_type] = self.calculate_metrics(trades)
        
        return metrics

class GUI:
    """GUI for displaying backtest results."""
    
    def __init__(self, master, metrics, trades):
        """Initialize the GUI.
        
        Args:
            master: Tkinter root window
            metrics: Performance metrics
            trades: List of trades
        """
        self.master = master
        self.metrics = metrics
        self.trades = trades
        
        master.title("Trading Log Analysis")
        master.geometry("1000x800")
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.summary_tab = ttk.Frame(self.notebook)
        self.trades_tab = ttk.Frame(self.notebook)
        self.charts_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.summary_tab, text="Summary")
        self.notebook.add(self.trades_tab, text="Trades")
        self.notebook.add(self.charts_tab, text="Charts")
        
        # Populate tabs
        self._create_summary_tab()
        self._create_trades_tab()
        self._create_charts_tab()
    
    def _create_summary_tab(self):
        """Create the summary tab."""
        # Create a header
        header = ttk.Label(self.summary_tab, text="Performance Summary by Prompt Type", font=("Arial", 16, "bold"))
        header.pack(pady=10)
        
        # Create a frame for the metrics table
        table_frame = ttk.Frame(self.summary_tab)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create table headers
        columns = ["Prompt Type", "Total Trades", "Wins", "Losses", "Win Rate", "Loss Rate", "Inconclusive", 
                   "Avg Profit %", "Avg Loss %", "Profit Factor"]
        
        for i, col in enumerate(columns):
            label = ttk.Label(table_frame, text=col, font=("Arial", 10, "bold"))
            label.grid(row=0, column=i, padx=5, pady=5, sticky="w")
        
        # Add data rows
        row = 1
        for prompt_type, metric in self.metrics.items():
            ttk.Label(table_frame, text=prompt_type).grid(row=row, column=0, padx=5, pady=2, sticky="w")
            ttk.Label(table_frame, text=str(metric["total_trades"])).grid(row=row, column=1, padx=5, pady=2)
            ttk.Label(table_frame, text=str(metric["wins"])).grid(row=row, column=2, padx=5, pady=2)
            ttk.Label(table_frame, text=str(metric["losses"])).grid(row=row, column=3, padx=5, pady=2)
            ttk.Label(table_frame, text=f"{metric['win_rate']:.2%}").grid(row=row, column=4, padx=5, pady=2)
            ttk.Label(table_frame, text=f"{metric['loss_rate']:.2%}").grid(row=row, column=5, padx=5, pady=2)
            ttk.Label(table_frame, text=f"{metric['inconclusive_rate']:.2%}").grid(row=row, column=6, padx=5, pady=2)
            ttk.Label(table_frame, text=f"{metric['avg_profit_pct']:.2f}%").grid(row=row, column=7, padx=5, pady=2)
            ttk.Label(table_frame, text=f"{metric['avg_loss_pct']:.2f}%").grid(row=row, column=8, padx=5, pady=2)
            ttk.Label(table_frame, text=f"{metric['profit_factor']:.2f}").grid(row=row, column=9, padx=5, pady=2)
            row += 1
    
    def _create_trades_tab(self):
        """Create the trades tab."""
        # Create a header
        header = ttk.Label(self.trades_tab, text="Individual Trades", font=("Arial", 16, "bold"))
        header.pack(pady=10)
        
        # Create frame for table
        table_frame = ttk.Frame(self.trades_tab)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create scrollbars
        y_scrollbar = ttk.Scrollbar(table_frame)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        x_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create table
        columns = ["Date", "Symbol", "Prompt Type", "Position", "Entry", "TP", "SL", "Outcome"]
        
        self.trades_table = ttk.Treeview(
            table_frame, 
            columns=columns,
            show="headings",
            yscrollcommand=y_scrollbar.set,
            xscrollcommand=x_scrollbar.set
        )
        
        # Configure scrollbars
        y_scrollbar.config(command=self.trades_table.yview)
        x_scrollbar.config(command=self.trades_table.xview)
        
        # Set column headings
        for col in columns:
            self.trades_table.heading(col, text=col)
            self.trades_table.column(col, width=100, anchor=tk.CENTER)
        
        # Insert data
        for trade in self.trades:
            position = trade.get("current_position", {}).get("position", 0)
            take_profit = trade.get("current_position", {}).get("take_profit", 0)
            stop_loss = trade.get("current_position", {}).get("stop_loss", 0)
            
            self.trades_table.insert("", tk.END, values=(
                trade.get("timestamp", "").strftime("%Y-%m-%d %H:%M") if isinstance(trade.get("timestamp"), datetime) else "",
                trade.get("symbol", ""),
                trade.get("prompt_type", ""),
                f"{position:.2f}",
                f"{trade.get('entry_price', 0):.2f}",
                f"{take_profit:.2f}",
                f"{stop_loss:.2f}",
                trade.get("outcome", "")
            ))
        
        self.trades_table.pack(fill=tk.BOTH, expand=True)
    
    def _create_charts_tab(self):
        """Create the charts tab."""
        # Create a header
        header = ttk.Label(self.charts_tab, text="Performance Charts", font=("Arial", 16, "bold"))
        header.pack(pady=10)
        
        # Create frame for charts
        charts_frame = ttk.Frame(self.charts_tab)
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Get color values for prompt types
        prompt_types = list(self.metrics.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        color_map = {pt: colors[i % len(colors)] for i, pt in enumerate(prompt_types)}
        
        # Create figure with subplots
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        
        # Win rate by prompt type
        ax1 = fig.add_subplot(221)
        win_rates = [metric["win_rate"] * 100 for metric in self.metrics.values()]
        bars = ax1.bar(prompt_types, win_rates, color=[color_map[pt] for pt in prompt_types])
        ax1.set_title("Win Rate by Prompt Type")
        ax1.set_ylabel("Win Rate (%)")
        ax1.set_ylim(0, 100)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', rotation=0
            )
        
        # Profit factor by prompt type
        ax2 = fig.add_subplot(222)
        profit_factors = [metric["profit_factor"] for metric in self.metrics.values()]
        bars = ax2.bar(prompt_types, profit_factors, color=[color_map[pt] for pt in prompt_types])
        ax2.set_title("Profit Factor by Prompt Type")
        ax2.set_ylabel("Profit Factor")
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', rotation=0
            )
        
        # Trade count by prompt type
        ax3 = fig.add_subplot(223)
        trade_counts = [metric["total_trades"] for metric in self.metrics.values()]
        bars = ax3.bar(prompt_types, trade_counts, color=[color_map[pt] for pt in prompt_types])
        ax3.set_title("Number of Trades by Prompt Type")
        ax3.set_ylabel("Trade Count")
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', rotation=0
            )
        
        # Average profit/loss by prompt type (only if there are trades)
        ax4 = fig.add_subplot(224)
        avg_profits = [metric["avg_profit_pct"] for metric in self.metrics.values()]
        avg_losses = [abs(metric["avg_loss_pct"]) for metric in self.metrics.values()]
        
        # Set width of bars
        bar_width = 0.35
        r1 = np.arange(len(prompt_types))
        r2 = [x + bar_width for x in r1]
        
        bars1 = ax4.bar(r1, avg_profits, width=bar_width, color='green', label='Avg Profit %')
        bars2 = ax4.bar(r2, avg_losses, width=bar_width, color='red', label='Avg Loss %')
        
        ax4.set_title("Average Profit/Loss by Prompt Type")
        ax4.set_ylabel("Percentage")
        ax4.set_xticks([r + bar_width/2 for r in range(len(prompt_types))])
        ax4.set_xticklabels(prompt_types)
        ax4.legend()
        
        # Add value labels on top of bars
        for bar in bars1:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2., height + 0.05,
                f'{height:.2f}%', ha='center', va='bottom', rotation=0
            )
            
        for bar in bars2:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2., height + 0.05,
                f'{height:.2f}%', ha='center', va='bottom', rotation=0
            )
        
        # Add spacing between subplots
        fig.tight_layout(pad=2.0)
        
        # Add the figure to the GUI
        canvas = FigureCanvasTkAgg(fig, master=charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

async def main():
    """Main function to run the log analyzer."""
    try:
        analyzer = LogAnalyzer()
        metrics = await analyzer.analyze_logs()
        
        # Display results
        print("\nPerformance Metrics by Prompt Type:")
        for prompt_type, metric in metrics.items():
            print(f"\n{prompt_type.upper()}:")
            print(f"  Total Trades: {metric['total_trades']}")
            print(f"  Win Rate: {metric['win_rate']:.2%}")
            print(f"  Loss Rate: {metric['loss_rate']:.2%}")
            print(f"  Inconclusive: {metric['inconclusive_rate']:.2%}")
            print(f"  Avg Profit: {metric['avg_profit_pct']:.2f}%")
            print(f"  Avg Loss: {metric['avg_loss_pct']:.2f}%")
            print(f"  Profit Factor: {metric['profit_factor']:.2f}")
        
        # Launch GUI
        root = tk.Tk()
        app = GUI(root, metrics, analyzer.trades)
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    # Set up asyncio for Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the main function
    asyncio.run(main()) 