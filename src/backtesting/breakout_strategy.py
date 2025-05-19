"""
Implementation of breakout trading strategy.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from .strategy import Strategy, MarketData, Signal

class BreakoutStrategy(Strategy):
    """
    Breakout trading strategy implementation.
    
    This strategy generates buy signals when price breaks above resistance levels
    and sell signals when price breaks below support levels, with confirmation
    from volume and volatility indicators.
    """
    
    def __init__(self, name: str = "Breakout", parameters: Dict[str, Any] = None):
        """
        Initialize breakout strategy with parameters.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
        """
        default_params = {
            "lookback_period": 20,        # Period for identifying support/resistance
            "channel_period": 20,         # Period for price channel calculation
            "volume_confirmation": True,   # Whether to require volume confirmation
            "volume_factor": 1.5,         # Volume must be this times the average
            "atr_period": 14,             # ATR period for volatility measurement
            "atr_multiplier": 0.5,        # Breakout must exceed this * ATR
            "false_breakout_filter": True, # Whether to filter false breakouts
            "confirmation_bars": 1,       # Bars needed to confirm breakout
            "stop_loss_atr": 2.0,         # Stop loss as multiple of ATR
            "take_profit_atr": 3.0,       # Take profit as multiple of ATR
            "trailing_stop": True,        # Whether to use trailing stop
            "trailing_stop_atr": 2.0      # Trailing stop as multiple of ATR
        }
        
        # Merge default parameters with provided parameters
        merged_params = default_params.copy()
        if parameters:
            merged_params.update(parameters)
            
        super().__init__(name, merged_params)
        
        # Initialize data storage
        self.price_history = []
        self.indicators = {}
        self.current_position = None  # "LONG", "SHORT", or None
        self.position_entry_price = None
        self.position_entry_time = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
        self.breakout_confirmation_count = 0
        self.potential_breakout = None  # "UP", "DOWN", or None
    
    def initialize(self) -> None:
        """Initialize strategy with parameters."""
        super().initialize()
        self.price_history = []
        self.indicators = {
            "upper_channel": [],
            "lower_channel": [],
            "atr": [],
            "volume_sma": []
        }
    
    def process_market_data(self, data: MarketData) -> None:
        """
        Process new market data.
        
        Args:
            data: Market data to process
        """
        # Store price data
        self.price_history.append({
            "timestamp": data.timestamp,
            "open": data.open,
            "high": data.high,
            "low": data.low,
            "close": data.close,
            "volume": data.volume
        })
        
        # Update indicators if we have enough data
        if len(self.price_history) >= max(
            self.parameters["lookback_period"],
            self.parameters["channel_period"],
            self.parameters["atr_period"]
        ):
            self._update_indicators()
        
        # Update position tracking
        if self.current_position == "LONG":
            if data.high > (self.highest_price_since_entry or 0):
                self.highest_price_since_entry = data.high
        elif self.current_position == "SHORT":
            if data.low < (self.lowest_price_since_entry or float('inf')) or self.lowest_price_since_entry is None:
                self.lowest_price_since_entry = data.low
    
    def _update_indicators(self) -> None:
        """Update technical indicators based on price history."""
        # Convert price history to DataFrame for easier calculation
        df = pd.DataFrame(self.price_history)
        
        # Calculate price channels (resistance and support)
        channel_period = self.parameters["channel_period"]
        if len(df) >= channel_period:
            upper_channel = df['high'].rolling(window=channel_period).max().iloc[-1]
            lower_channel = df['low'].rolling(window=channel_period).min().iloc[-1]
            
            self.indicators["upper_channel"].append(upper_channel)
            self.indicators["lower_channel"].append(lower_channel)
        
        # Calculate ATR (Average True Range)
        atr_period = self.parameters["atr_period"]
        if len(df) > atr_period:
            high = df['high']
            low = df['low']
            close = df['close'].shift()
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=atr_period).mean().iloc[-1]
            
            self.indicators["atr"].append(atr)
        
        # Calculate Volume SMA
        lookback_period = self.parameters["lookback_period"]
        if len(df) >= lookback_period:
            volume_sma = df['volume'].rolling(window=lookback_period).mean().iloc[-1]
            self.indicators["volume_sma"].append(volume_sma)
    
    def generate_signals(self) -> List[Signal]:
        """
        Generate trading signals based on processed data.
        
        Returns:
            List of trading signals
        """
        signals = []
        
        # Need enough data to generate signals
        if (not self.price_history or 
            len(self.indicators.get("upper_channel", [])) == 0 or 
            len(self.indicators.get("atr", [])) == 0):
            return signals
        
        current_price = self.price_history[-1]["close"]
        current_time = self.price_history[-1]["timestamp"]
        
        # Check for exit signals first (if in a position)
        if self.current_position is not None:
            exit_signal = self._check_exit_signals(current_price, current_time)
            if exit_signal:
                signals.append(exit_signal)
                
                # Reset position tracking
                self.current_position = None
                self.position_entry_price = None
                self.position_entry_time = None
                self.highest_price_since_entry = None
                self.lowest_price_since_entry = None
                
                # Return early - don't enter a new position in the same bar
                return signals
        
        # Check for entry signals (if not in a position)
        if self.current_position is None:
            entry_signal = self._check_entry_signals(current_price, current_time)
            if entry_signal:
                signals.append(entry_signal)
                
                # Update position tracking
                self.current_position = "LONG" if entry_signal.signal_type == Signal.BUY else "SHORT"
                self.position_entry_price = current_price
                self.position_entry_time = current_time
                self.highest_price_since_entry = current_price
                self.lowest_price_since_entry = current_price
                
                # Reset breakout tracking
                self.potential_breakout = None
                self.breakout_confirmation_count = 0
        
        return signals
    
    def _check_entry_signals(self, current_price: float, current_time: datetime) -> Optional[Signal]:
        """
        Check for entry signals.
        
        Args:
            current_price: Current price
            current_time: Current timestamp
            
        Returns:
            Entry signal or None
        """
        # Get latest indicator values
        upper_channel = self.indicators["upper_channel"][-1] if self.indicators["upper_channel"] else float('inf')
        lower_channel = self.indicators["lower_channel"][-1] if self.indicators["lower_channel"] else 0
        atr = self.indicators["atr"][-1] if self.indicators["atr"] else (current_price * 0.01)  # Default to 1% of price
        
        # Volume confirmation
        current_volume = self.price_history[-1]["volume"]
        volume_sma = self.indicators["volume_sma"][-1] if self.indicators["volume_sma"] else 0
        volume_confirmation = not self.parameters["volume_confirmation"] or (
            current_volume > (volume_sma * self.parameters["volume_factor"])
        )
        
        # Minimum breakout size
        min_breakout_size = atr * self.parameters["atr_multiplier"]
        
        # Check for potential breakouts
        if self.potential_breakout is None:
            # Upward breakout
            if current_price > upper_channel and (current_price - upper_channel) > min_breakout_size:
                self.potential_breakout = "UP"
                self.breakout_confirmation_count = 1
            # Downward breakout
            elif current_price < lower_channel and (lower_channel - current_price) > min_breakout_size:
                self.potential_breakout = "DOWN"
                self.breakout_confirmation_count = 1
        else:
            # Continue tracking existing potential breakout
            if self.potential_breakout == "UP" and current_price > upper_channel:
                self.breakout_confirmation_count += 1
            elif self.potential_breakout == "DOWN" and current_price < lower_channel:
                self.breakout_confirmation_count += 1
            else:
                # Reset if breakout fails
                self.potential_breakout = None
                self.breakout_confirmation_count = 0
        
        # Check if we have enough confirmation bars
        confirmed_breakout = (
            self.potential_breakout is not None and 
            self.breakout_confirmation_count >= self.parameters["confirmation_bars"]
        )
        
        # Generate signals for confirmed breakouts
        if confirmed_breakout and volume_confirmation:
            if self.potential_breakout == "UP":
                return Signal(
                    symbol=self.price_history[-1].get("symbol", "UNKNOWN"),
                    signal_type=Signal.BUY,
                    timestamp=current_time,
                    price=current_price,
                    strength=min(1.0, self.breakout_confirmation_count / 5),  # Normalize strength
                    metadata={
                        "breakout_level": upper_channel,
                        "atr": atr,
                        "confirmation_bars": self.breakout_confirmation_count,
                        "reason": "Upward breakout"
                    }
                )
            elif self.potential_breakout == "DOWN":
                return Signal(
                    symbol=self.price_history[-1].get("symbol", "UNKNOWN"),
                    signal_type=Signal.SELL,
                    timestamp=current_time,
                    price=current_price,
                    strength=min(1.0, self.breakout_confirmation_count / 5),  # Normalize strength
                    metadata={
                        "breakout_level": lower_channel,
                        "atr": atr,
                        "confirmation_bars": self.breakout_confirmation_count,
                        "reason": "Downward breakout"
                    }
                )
        
        return None
    
    def _check_exit_signals(self, current_price: float, current_time: datetime) -> Optional[Signal]:
        """
        Check for exit signals.
        
        Args:
            current_price: Current price
            current_time: Current timestamp
            
        Returns:
            Exit signal or None
        """
        if not self.current_position or not self.position_entry_price:
            return None
        
        # Get latest indicator values
        atr = self.indicators["atr"][-1] if self.indicators["atr"] else (current_price * 0.01)  # Default to 1% of price
        
        # Variables for signal generation
        exit_reason = None
        
        # Check for long position exit
        if self.current_position == "LONG":
            # Initial stop loss
            stop_loss = self.position_entry_price - (atr * self.parameters["stop_loss_atr"])
            
            # Trailing stop if enabled
            if (self.parameters["trailing_stop"] and 
                self.highest_price_since_entry is not None):
                trailing_stop = self.highest_price_since_entry - (atr * self.parameters["trailing_stop_atr"])
                stop_loss = max(stop_loss, trailing_stop)
            
            # Take profit
            take_profit = self.position_entry_price + (atr * self.parameters["take_profit_atr"])
            
            # Check exit conditions
            if current_price <= stop_loss:
                exit_reason = "Stop loss"
            elif current_price >= take_profit:
                exit_reason = "Take profit"
        
        # Check for short position exit
        elif self.current_position == "SHORT":
            # Initial stop loss
            stop_loss = self.position_entry_price + (atr * self.parameters["stop_loss_atr"])
            
            # Trailing stop if enabled
            if (self.parameters["trailing_stop"] and 
                self.lowest_price_since_entry is not None):
                trailing_stop = self.lowest_price_since_entry + (atr * self.parameters["trailing_stop_atr"])
                stop_loss = min(stop_loss, trailing_stop)
            
            # Take profit
            take_profit = self.position_entry_price - (atr * self.parameters["take_profit_atr"])
            
            # Check exit conditions
            if current_price >= stop_loss:
                exit_reason = "Stop loss"
            elif current_price <= take_profit:
                exit_reason = "Take profit"
        
        # Generate exit signal if we have a reason
        if exit_reason:
            signal_type = Signal.CLOSE_LONG if self.current_position == "LONG" else Signal.CLOSE_SHORT
            
            return Signal(
                symbol=self.price_history[-1].get("symbol", "UNKNOWN"),
                signal_type=signal_type,
                timestamp=current_time,
                price=current_price,
                metadata={
                    "entry_price": self.position_entry_price,
                    "position_duration": (current_time - self.position_entry_time).total_seconds() / 3600,  # hours
                    "reason": exit_reason
                }
            )
        
        return None
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Return strategy performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        # This would typically be populated by the backtesting engine
        # Here we just return a placeholder
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0
        }
