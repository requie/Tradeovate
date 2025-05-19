"""
Main entry point for the Tradeovate trading bot.
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime

from src.config import ConfigManager
from src.trading.engine import TradingEngine

# Set up logging
def setup_logging(log_level, log_file):
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    """Main entry point for the trading bot."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tradeovate Algorithmic Trading Bot')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--log-file', default='logs/trading_bot.log', help='Path to log file')
    parser.add_argument('--version', action='version', version='Tradeovate Trading Bot v1.0.0')
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Log startup information
        logger.info("Starting Tradeovate Trading Bot")
        logger.info(f"Configuration file: {args.config}")
        logger.info(f"Log level: {args.log_level}")
        logger.info(f"Log file: {args.log_file}")
        
        # Load configuration
        config_manager = ConfigManager(args.config)
        
        # Validate configuration
        errors = config_manager.validate()
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            logger.error("Invalid configuration, exiting")
            return 1
        
        # Create trading engine
        engine = TradingEngine(args.config)
        
        # Start trading engine
        engine.start()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping")
        finally:
            # Stop trading engine
            engine.stop()
        
        logger.info("Trading bot stopped")
        return 0
        
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
