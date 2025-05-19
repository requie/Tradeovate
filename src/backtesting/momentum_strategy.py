"""
Implementation of momentum trading strategy.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from .strategy import Strategy, MarketData, Signal

class MomentumStrategy(Strategy):
    """
    Momentum trading strategy implementation.
    
    This strategy generates buy signals when price momentum is positive
    and sell signals when momentum is negative, based on the rate of change
    over a specified period and confirmation from other indicators.
    """
    
    def __init__(self, name: str = "Momentum", parameters: Dict[str, Any] = None):
        """
        Initialize momentum strategy with parameters.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
        """
        default_params = {
            "roc_period": 14,           # Rate of Change period
            "rsi_period": 14,           # RSI period
            "rsi_overbought": 70,       # RSI overbought threshold
            "rsi_oversold": 30,         # RSI oversold threshold
            "macd_fast": 12,            # MACD fast period
            "macd_slow": 26,            # MACD slow period
            "macd_signal": 9,           # MACD signal period
            "entry_threshold": 0.5,     # Momentum threshold for entry
            "exit_threshold": -0.2,     # Momentum threshold for exit
            "stop_loss_pct": 0.02,      # Stop loss percentage
            "take_profit_pct": 0.05,    # Take profit percentage
            "use_trailing_stop": True,  # Whether to use trailing stop
            "trailing_stop_pct": 0.015  # Trailing stop percentage
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
    
    def initialize(self) -> None:
        """Initialize strategy with parameters."""
        super().initialize()
        self.price_history = []
        self.indicators = {
            "roc": [],
            "rsi": [],
            "macd": [],
            "macd_signal": [],
            "macd_histogram": []
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
            self.parameters["roc_period"],
            self.parameters["rsi_period"],
            self.parameters["macd_slow"] + self.parameters["macd_signal"]
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
        
        # Calculate Rate of Change (ROC)
        roc_period = self.parameters["roc_period"]
        if len(df) > roc_period:
            roc = ((df['close'].iloc[-1] - df['close'].iloc[-roc_period-1]) / 
                   df['close'].iloc[-roc_period-1] * 100)
            self.indicators["roc"].append(roc)
        
        # Calculate RSI
        rsi_period = self.parameters["rsi_period"]
        if len(df) > rsi_period:
            delta = df['close'].diff().dropna()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            
            if loss.iloc[-1] == 0:
                rsi = 100
            else:
                rs = gain.iloc[-1] / loss.iloc[-1]
                rsi = 100 - (100 / (1 + rs))
            
            self.indicators["rsi"].append(rsi)
        
        # Calculate MACD
        fast_period = self.parameters["macd_fast"]
        slow_period = self.parameters["macd_slow"]
        signal_period = self.parameters["macd_signal"]
        
        if len(df) > slow_period + signal_period:
            # Calculate EMAs
            ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line (EMA of MACD line)
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # Histogram
            histogram = macd_line - signal_line
            
            self.indicators["macd"].append(macd_line.iloc[-1])
            self.indicators["macd_signal"].append(signal_line.iloc[-1])
            self.indicators["macd_histogram"].append(histogram.iloc[-1])
    
    def generate_signals(self) -> List[Signal]:
        """
        Generate trading signals based on processed data.
        
        Returns:
            List of trading signals
        """
        signals = []
        
        # Need enough data to generate signals
        if (not self.price_history or 
            len(self.indicators.get("roc", [])) == 0 or 
            len(self.indicators.get("rsi", [])) == 0 or 
            len(self.indicators.get("macd", [])) == 0):
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
        roc = self.indicators["roc"][-1] if self.indicators["roc"] else 0
        rsi = self.indicators["rsi"][-1] if self.indicators["rsi"] else 50
        macd = self.indicators["macd"][-1] if self.indicators["macd"] else 0
        macd_signal = self.indicators["macd_signal"][-1] if self.indicators["macd_signal"] else 0
        macd_histogram = self.indicators["macd_histogram"][-1] if self.indicators["macd_histogram"] else 0
        
        # Long entry conditions
        if (roc > self.parameters["entry_threshold"] and 
            rsi < 70 and  # Not overbought
            macd > macd_signal and  # MACD crossover
            macd_histogram > 0):
            
            return Signal(
                symbol=self.price_history[-1].get("symbol", "UNKNOWN"),
                signal_type=Signal.BUY,
                timestamp=current_time,
                price=current_price,
                strength=min(1.0, roc / 10),  # Normalize strength
                metadata={
                    "roc": roc,
                    "rsi": rsi,
                    "macd": macd,
                    "reason": "Momentum long entry"
                }
            )
        
        # Short entry conditions
        elif (roc < -self.parameters["entry_threshold"] and 
              rsi > 30 and  # Not oversold
              macd < macd_signal and  # MACD crossover
              macd_histogram < 0):
            
            return Signal(
                symbol=self.price_history[-1].get("symbol", "UNKNOWN"),
                signal_type=Signal.SELL,
                timestamp=current_time,
                price=current_price,
                strength=min(1.0, abs(roc) / 10),  # Normalize strength
                metadata={
                    "roc": roc,
                    "rsi": rsi,
                    "macd": macd,
                    "reason": "Momentum short entry"
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
        roc = self.indicators["roc"][-1] if self.indicators["roc"] else 0
        
        # Variables for signal generation
        exit_reason = None
        
        # Check for long position exit
        if self.current_position == "LONG":
            # Stop loss
            stop_price = self.position_entry_price * (1 - self.parameters["stop_loss_pct"])
            
            # Trailing stop if enabled
            if (self.parameters["use_trailing_stop"] and 
                self.highest_price_since_entry is not None):
                trailing_stop = self.highest_price_since_entry * (1 - self.parameters["trailing_stop_pct"])
                stop_price = max(stop_price, trailing_stop)
            
            # Take profit
            take_profit = self.position_entry_price * (1 + self.parameters["take_profit_pct"])
            
            # Check exit conditions
            if current_price <= stop_price:
                exit_reason = "Stop loss"
            elif current_price >= take_profit:
                exit_reason = "Take profit"
            elif roc < self.parameters["exit_threshold"]:
                exit_reason = "Momentum reversal"
        
        # Check for short position exit
        elif self.current_position == "SHORT":
            # Stop loss
            stop_price = self.position_entry_price * (1 + self.parameters["stop_loss_pct"])
            
            # Trailing stop if enabled
            if (self.parameters["use_trailing_stop"] and 
                self.lowest_price_since_entry is not None):
                trailing_stop = self.lowest_price_since_entry * (1 + self.parameters["trailing_stop_pct"])
                stop_price = min(stop_price, trailing_stop)
            
            # Take profit
            take_profit = self.position_entry_price * (1 - self.parameters["take_profit_pct"])
            
            # Check exit conditions
            if current_price >= stop_price:
                exit_reason = "Stop loss"
            elif current_price <= take_profit:
                exit_reason = "Take profit"
            elif roc > -self.parameters["exit_threshold"]:
                exit_reason = "Momentum reversal"
        
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
