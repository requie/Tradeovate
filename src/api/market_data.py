"""
Market data handling module for Tradeovate API.
"""
from typing import Dict, Any, List, Optional, Callable, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import threading
import queue
import time

from .client import TradeovateApiClient, TradeovateApiError

logger = logging.getLogger(__name__)

class MarketData:
    """
    Class for handling market data from Tradeovate API.
    
    This class manages market data subscriptions, processes incoming data,
    and provides methods for accessing historical and real-time data.
    """
    
    def __init__(self, api_client: TradeovateApiClient):
        """
        Initialize the market data handler.
        
        Args:
            api_client: Tradeovate API client
        """
        self.api_client = api_client
        self.subscriptions = {}  # symbol -> callback
        self.data_buffers = {}   # symbol -> DataFrame
        self.last_update = {}    # symbol -> timestamp
        
        # For handling data in a separate thread
        self.data_queue = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        
        # Maximum buffer size (number of bars to keep in memory)
        self.max_buffer_size = 1000
    
    def start(self) -> None:
        """Start the market data processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._process_data_queue)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Market data processing thread started")
    
    def stop(self) -> None:
        """Stop the market data processing thread."""
        self.is_processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            logger.info("Market data processing thread stopped")
    
    def subscribe(self, 
                 symbol: str, 
                 callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 chart_type: str = "Chart",
                 timeframe: str = "1 min") -> None:
        """
        Subscribe to market data for a symbol.
        
        Args:
            symbol: Trading symbol
            callback: Optional callback function for real-time updates
            chart_type: Chart type ("Tick", "DOM", "Chart")
            timeframe: Timeframe for chart data
        """
        # Initialize data buffer for this symbol if not exists
        if symbol not in self.data_buffers:
            self.data_buffers[symbol] = pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            self.data_buffers[symbol].set_index("timestamp", inplace=True)
        
        # Store callback if provided
        if callback:
            self.subscriptions[symbol] = callback
        
        # Subscribe to market data
        self.api_client.subscribe_market_data(
            symbol=symbol,
            callback=self._handle_market_data,
            chart_type=chart_type,
            timeframe=timeframe
        )
        
        logger.info(f"Subscribed to {symbol} market data with {timeframe} timeframe")
    
    def unsubscribe(self, symbol: str) -> None:
        """
        Unsubscribe from market data for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        if symbol in self.subscriptions:
            del self.subscriptions[symbol]
        
        # Note: Tradeovate API doesn't have a specific unsubscribe method,
        # so we just stop processing the data
        
        logger.info(f"Unsubscribed from {symbol} market data")
    
    def get_historical_data(self, 
                           symbol: str, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_time: Start time for data (None for all available)
            end_time: End time for data (None for all available)
            limit: Maximum number of bars to return
            
        Returns:
            DataFrame with historical data
        """
        # Check if we have data for this symbol
        if symbol not in self.data_buffers:
            logger.warning(f"No historical data available for {symbol}")
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            ).set_index("timestamp")
        
        # Get data from buffer
        data = self.data_buffers[symbol].copy()
        
        # Apply time filters if provided
        if start_time:
            data = data[data.index >= start_time]
        
        if end_time:
            data = data[data.index <= end_time]
        
        # Apply limit if provided
        if limit and len(data) > limit:
            data = data.iloc[-limit:]
        
        return data
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest price or None if not available
        """
        if symbol in self.data_buffers and not self.data_buffers[symbol].empty:
            return self.data_buffers[symbol]["close"].iloc[-1]
        return None
    
    def _handle_market_data(self, data: Dict[str, Any]) -> None:
        """
        Handle incoming market data from WebSocket.
        
        Args:
            data: Market data message
        """
        # Add to processing queue
        self.data_queue.put(data)
    
    def _process_data_queue(self) -> None:
        """Process market data queue in a separate thread."""
        while self.is_processing:
            try:
                # Get data from queue with timeout to allow checking is_processing
                try:
                    data = self.data_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the data
                self._process_market_data(data)
                
                # Mark task as done
                self.data_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing market data: {str(e)}")
    
    def _process_market_data(self, data: Dict[str, Any]) -> None:
        """
        Process a market data message.
        
        Args:
            data: Market data message
        """
        try:
            # Extract chart data
            if "chart" in data and "symbol" in data["chart"]:
                symbol = data["chart"]["symbol"]
                
                # Process based on chart type
                if "bars" in data["chart"]:
                    self._process_bar_data(symbol, data["chart"]["bars"])
                elif "dom" in data["chart"]:
                    self._process_dom_data(symbol, data["chart"]["dom"])
                elif "histogram" in data["chart"]:
                    self._process_histogram_data(symbol, data["chart"]["histogram"])
                
                # Call symbol-specific callback if registered
                if symbol in self.subscriptions and self.subscriptions[symbol]:
                    self.subscriptions[symbol](data)
        
        except Exception as e:
            logger.error(f"Error processing market data message: {str(e)}")
    
    def _process_bar_data(self, symbol: str, bars: List[Dict[str, Any]]) -> None:
        """
        Process bar data.
        
        Args:
            symbol: Trading symbol
            bars: List of bar data
        """
        if not bars:
            return
        
        # Create DataFrame from bars
        new_data = []
        
        for bar in bars:
            timestamp = datetime.fromtimestamp(bar.get("timestamp") / 1000)  # Convert ms to seconds
            
            new_data.append({
                "timestamp": timestamp,
                "open": bar.get("open", 0.0),
                "high": bar.get("high", 0.0),
                "low": bar.get("low", 0.0),
                "close": bar.get("close", 0.0),
                "volume": bar.get("volume", 0)
            })
        
        if new_data:
            # Convert to DataFrame
            df = pd.DataFrame(new_data)
            df.set_index("timestamp", inplace=True)
            
            # Update data buffer
            if symbol in self.data_buffers:
                # Append new data, avoiding duplicates
                self.data_buffers[symbol] = pd.concat([self.data_buffers[symbol], df])
                self.data_buffers[symbol] = self.data_buffers[symbol][~self.data_buffers[symbol].index.duplicated(keep='last')]
                
                # Sort by timestamp
                self.data_buffers[symbol].sort_index(inplace=True)
                
                # Limit buffer size
                if len(self.data_buffers[symbol]) > self.max_buffer_size:
                    self.data_buffers[symbol] = self.data_buffers[symbol].iloc[-self.max_buffer_size:]
            else:
                self.data_buffers[symbol] = df
            
            # Update last update time
            self.last_update[symbol] = datetime.now()
    
    def _process_dom_data(self, symbol: str, dom: Dict[str, Any]) -> None:
        """
        Process depth of market (DOM) data.
        
        Args:
            symbol: Trading symbol
            dom: DOM data
        """
        # DOM data processing would be implemented here
        # This is a placeholder for future implementation
        pass
    
    def _process_histogram_data(self, symbol: str, histogram: Dict[str, Any]) -> None:
        """
        Process histogram data.
        
        Args:
            symbol: Trading symbol
            histogram: Histogram data
        """
        # Histogram data processing would be implemented here
        # This is a placeholder for future implementation
        pass
