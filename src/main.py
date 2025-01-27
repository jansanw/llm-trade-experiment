import os
import argparse
from datetime import datetime, timedelta
import asyncio
import platform
from dotenv import load_dotenv
import logging

from src.llm.deepseek_provider import DeepSeekProvider
from src.prompts.generators import PromptV0, PromptFVG
from src.bot.trading_bot import TradingBot
from src.backtest.engine import BacktestEngine
from src.dashboard.app import Dashboard

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Set up logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # Console handler with detailed formatting
            logging.StreamHandler(),
            # File handler for persistent logs
            logging.FileHandler(f'logs/trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('src.llm.deepseek_provider').setLevel(logging.DEBUG)
    logging.getLogger('urllib3').setLevel(logging.INFO)  # Reduce noise from HTTP client
    logging.getLogger('asyncio').setLevel(logging.INFO)  # Reduce noise from asyncio
    logging.getLogger('yfinance').setLevel(logging.INFO)  # Reduce noise from yfinance
    logging.getLogger('peewee').setLevel(logging.INFO)  # Reduce noise from peewee

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
        default="MNQ",
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
        choices=['v0', 'fvg'],
        default='fvg',
        help='Type of prompt generator to use'
    )
    return parser.parse_args()

async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_args()
    
    # Initialize LLM provider
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")
    
    # Select prompt generator based on argument
    prompt_generator = PromptFVG() if args.prompt_type == 'fvg' else PromptV0()
        
    llm = DeepSeekProvider(
        api_key=api_key,
        dry_run=args.dry_run,
        prompt_generator=prompt_generator
    )
    
    # Initialize trading bot
    bot = TradingBot(
        llm_provider=llm,
        symbol=args.symbol,
        update_interval=args.update_interval
    )
    
    if args.mode == "oneshot":
        # Get single analysis
        decision = await bot.get_trading_decision()
        print("\nTrading Decision:")
        print("-" * 50)
        print(f"Position: {'LONG' if decision['position'] > 0 else 'SHORT' if decision['position'] < 0 else 'NEUTRAL'}")
        print(f"Size: {abs(decision['position']):.2f}")
        print(f"Confidence: {decision['confidence']:.1%}")
        print(f"\nReasoning:\n{decision['reasoning']}")
        print("-" * 50)
        return
        
    if args.mode == "live":
        # Run live trading
        await bot.run_live()
        
    elif args.mode == "backtest":
        if not args.start_date or not args.end_date:
            print("Error: start-date and end-date required for backtest mode")
            return
            
        # Run backtest
        engine = BacktestEngine(
            bot=bot,
            start_date=args.start_date,
            end_date=args.end_date
        )
        results = await engine.run()
        print(results)
        
    else:  # dashboard mode
        # Start dashboard
        dashboard = Dashboard(bot)
        dashboard.run(debug=True)

if __name__ == "__main__":
    setup_logging()
    # Set event loop policy for Windows
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main()) 