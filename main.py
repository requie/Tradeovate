"""
Main entry point for the Tradeovate trading bot with FastAPI integration.
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import ConfigManager
from src.trading.engine import TradingEngine

# Create FastAPI app
app = FastAPI(title="Tradeovate Trading Bot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Frontend dev server HTTP
        "https://localhost:5173"  # Frontend dev server HTTPS
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
trading_engine: Optional[TradingEngine] = None
config_manager: Optional[ConfigManager] = None

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

# API Models
class BotStatus(BaseModel):
    status: str

# API Routes
@app.get("/api/stats")
async def get_stats():
    """Get current trading bot statistics."""
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")
    
    return {
        "totalTrades": trading_engine.get_total_trades(),
        "successRate": trading_engine.get_success_rate(),
        "totalProfit": trading_engine.get_total_profit(),
        "activePositions": trading_engine.get_active_positions()
    }

@app.post("/api/bot/status")
async def update_bot_status(status: BotStatus):
    """Update bot running status."""
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")
    
    if status.status == "running":
        trading_engine.start()
    elif status.status == "stopped":
        trading_engine.stop()
    else:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    return {"status": status.status}

def main():
    """Main entry point for the trading bot."""
    global trading_engine, config_manager
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tradeovate Algorithmic Trading Bot')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--log-file', default='logs/trading_bot.log', help='Path to log file')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API server')
    parser.add_argument('--ssl-keyfile', help='Path to SSL key file')
    parser.add_argument('--ssl-certfile', help='Path to SSL certificate file')
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
        trading_engine = TradingEngine(args.config)
        
        # Check if SSL files are provided and exist
        use_ssl = False
        if args.ssl_keyfile and args.ssl_certfile:
            if os.path.exists(args.ssl_keyfile) and os.path.exists(args.ssl_certfile):
                use_ssl = True
                logger.info("SSL certificates found, starting HTTPS server")
            else:
                logger.warning("SSL certificates not found, falling back to HTTP")
        else:
            logger.info("SSL certificates not provided, using HTTP")
        
        # Start the server with or without SSL
        server_config = {
            "app": app,
            "host": "0.0.0.0",
            "port": args.port
        }
        
        if use_ssl:
            server_config.update({
                "ssl_keyfile": args.ssl_keyfile,
                "ssl_certfile": args.ssl_certfile
            })
            logger.info(f"Starting HTTPS API server on port {args.port}")
        else:
            logger.info(f"Starting HTTP API server on port {args.port}")
        
        uvicorn.run(**server_config)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())