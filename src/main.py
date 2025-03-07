import os
import argparse
from datetime import datetime, timedelta
import asyncio
import platform
from dotenv import load_dotenv
import logging

from src.bot.trading_bot import TradingBot
from src.llm.deepseek_provider import DeepSeekProvider
from src.llm.mock_provider import MockProvider
from src.data.market_data import MarketDataFetcher
from src.backtest.engine import BacktestEngine
from src.dashboard.app import Dashboard
from src.prompts.generators import PromptV0, PromptFVG, PromptRaw, PromptRawUniform
from src.utils.logging import setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["live", "backtest", "dashboard", "oneshot"],
        default="dashboard",
        help="Operating mode"
    )
    parser.add_argument(
        "--symbol",
        default="SPY",
        help="Trading symbol"
    )
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="Start date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="End date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=60,
        help="Update interval in seconds for live trading"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only log the prompt without making API calls'
    )
    parser.add_argument(
        '--prompt-type',
        choices=['v0', 'fvg', 'raw', 'uniform'],
        default='fvg',
        help='Type of prompt generator to use'
    )
    parser.add_argument(
        '--provider',
        choices=['deepseek', 'mock'],
        default='deepseek',
        help='LLM provider to use'
    )
    parser.add_argument(
        '--mock-behavior',
        choices=['trend_follower', 'mean_reverter', 'random'],
        default='trend_follower',
        help='Behavior mode for mock provider'
    )
    return parser.parse_args()

async def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize logging and market data fetcher
    setup_logging()
    data_fetcher = MarketDataFetcher(symbol=args.symbol)
    
    # Initialize LLM provider
    if args.provider == "mock":
        llm = MockProvider(behavior=args.mock_behavior)
    else:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        llm = DeepSeekProvider(api_key=api_key, dry_run=args.dry_run)
    
    # Initialize trading bot
    bot = TradingBot(
        symbol=args.symbol,
        data_fetcher=data_fetcher,
        llm=llm,
        prompt_type=args.prompt_type
    )
    
    if args.mode == "oneshot":
        # Get trading decision
        try:
            decision = await bot.get_trading_decision()
            print(f"\nTrading Decision:")
            
            # Display daily bias
            if 'daily_bias' in decision:
                daily = decision['daily_bias']
                print(f"\nDaily Bias:")
                print(f"Direction: {'BULLISH' if daily['direction'] > 0 else 'BEARISH' if daily['direction'] < 0 else 'NEUTRAL'}")
                print(f"Confidence: {daily['confidence']*100:.1f}%")
                if daily['key_levels']:
                    print("Key Levels to Watch:")
                    for level in daily['key_levels']:
                        print(f"- {level}")
            
            # Display current position
            if 'current_position' in decision:
                pos = decision['current_position']
                print(f"\nCurrent Position:")
                print(f"Position: {'LONG' if pos['position'] > 0 else 'SHORT' if pos['position'] < 0 else 'NEUTRAL'}")
                print(f"Size: {abs(pos['position']):.2f}")
                print(f"Confidence: {pos['confidence']*100:.1f}%")
                if pos.get('take_profit'):
                    print(f"Take-Profit: {pos['take_profit']:.2f}")
                if pos.get('stop_loss'):
                    print(f"Stop-Loss: {pos['stop_loss']:.2f}")
            
            print(f"\nReasoning:\n{decision['reasoning']}")
            
        except Exception as e:
            print(f"Error getting trading decision: {str(e)}")
            return
            
    elif args.mode == "backtest":
        # Run backtest
        try:
            engine = BacktestEngine(
                bot=bot,
                start_date=args.start_date,
                end_date=args.end_date
            )
            await engine.run()
            
        except Exception as e:
            print(f"Error running backtest: {str(e)}")
            return

if __name__ == "__main__":
    asyncio.run(main()) 