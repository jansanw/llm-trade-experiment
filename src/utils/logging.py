import os
import logging
from datetime import datetime

def setup_logging():
    """Configure logging for the application."""
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
    logging.getLogger('polygon').setLevel(logging.INFO)  # Reduce noise from polygon
    logging.getLogger('peewee').setLevel(logging.INFO)  # Reduce noise from peewee 