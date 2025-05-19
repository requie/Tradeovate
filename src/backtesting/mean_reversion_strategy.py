"""
Implementation of mean reversion trading strategy.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from .strategy import Strategy, MarketData, Signal

class MeanReversionStrategy(Strategy):
    """
    Mean Reversion trading strategy implementation.
    
    This strategy generates buy signals when price is significantly below its mean
    and sell signals when price is significantly above its mean, using Bollinger Bands
    and other statistical measures to identify overbought and oversold conditions.
    """
    
    def __init__(self, name: str = "MeanReversion", parameters: Dict[str, Any] = None):
        """
        Initialize mean reversion strategy with parameters.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
        """
        default_params = {
            "lookback_period": 20,       # Period for calculating mean and std dev
            "entry_std_dev": 2.0,        # Standard deviations for entry
            "exit_std_dev": 0.5,         # Standard deviations for exit
            "rsi_period": 14,            # RSI period
            "rsi_overbought": 70,        # RSI overbought threshold
            "rsi_oversold": 30,          # RSI oversold threshold
            "volume_factor": 1.5,        # Volume confirmation factor
            "stop_loss_pct": 0.02,       # Stop loss percentage
            "take_profit_pct": 0.03,     # Take profit percentage
            "max_holding_periods": 10,   # Maximum holding periods
            "use_atr_stops": True,       # Whether to use ATR-based stops
            "atr_period": 14,            # ATR period
            "atr_multiplier": 3.0        # ATR multiplier for stops
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
        self.position_entry_index = None
    
    def initialize(self) -> None:
        """Initialize strategy with parameters."""
        super().initialize()
        self.price_history = []
        self.indicators = {
            "sma": [],
            "upper_band": [],
            "lower_band": [],
            "z_score": [],
            "rsi": [],
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
            self.parameters["rsi_period"],
            self.parameters["atr_period"]
        ):
            self._update_indicators()
    
    def _update_indicators(self) -> None:
        """Update technical indicators based on price history."""
        # Convert price history to DataFrame for easier calculation
        df = pd.DataFrame(self.price_history)
        
        # Calculate Simple Moving Average (SMA)
        lookback_period = self.parameters["lookback_period"]
        if len(df) >= lookback_period:
            sma = df['close'].rolling(window=lookback_period).mean().iloc[-1]
            std_dev = df['close'].rolling(window=lookback_period).std().iloc[-1]
            
            # Bollinger Bands
            upper_band = sma + (std_dev * self.parameters["entry_std_dev"])
            lower_band = sma - (std_dev * self.parameters["entry_std_dev"])
            
            # Z-score (how many standard deviations from mean)
            z_score = (df['close'].iloc[-1] - sma) / std_dev if std_dev > 0 else 0
            
            self.indicators["sma"].append(sma)
            self.indicators["upper_band"].append(upper_band)
            self.indicators["lower_band"].append(lower_band)
            self.indicators["z_score"].append(z_score)
        
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
        
        # Calculate ATR (Average True Range)
        atr_period = self.parameters["atr_period"]
        if len(df) > atr_period:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=atr_period).mean().iloc[-1]
            
            self.indicators["atr"].append(atr)
        
        # Calculate Volume SMA
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
            len(self.indicators.get("sma", [])) == 0 or 
            len(self.indicators.get("rsi", [])) == 0):
            return signals
        
        current_price = self.price_history[-1]["close"]
        current_time = self.price_history[-1]["timestamp"]
        current_index = len(self.price_history) - 1
        
        # Check for exit signals first (if in a position)
        if self.current_position is not None:
            exit_signal = self._check_exit_signals(current_price, current_time, current_index)
            if exit_signal:
                signals.append(exit_signal)
                
                # Reset position tracking
                self.current_position = None
                self.position_entry_price = None
                self.position_entry_time = None
                self.position_entry_index = None
                
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
                self.position_entry_index = current_index
        
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
        z_score = self.indicators["z_score"][-1] if self.indicators["z_score"] else 0
        rsi = self.indicators["rsi"][-1] if self.indicators["rsi"] else 50
        lower_band = self.indicators["lower_band"][-1] if self.indicators["lower_band"] else 0
        upper_band = self.indicators["upper_band"][-1] if self.indicators["upper_band"] else float('inf')
        
        # Volume confirmation
        current_volume = self.price_history[-1]["volume"]
        volume_sma = self.indicators["volume_sma"][-1] if self.indicators["volume_sma"] else 0
        volume_confirmation = current_volume > (volume_sma * self.parameters["volume_factor"])
        
        # Long entry conditions (price below lower band, oversold RSI)
        if (current_price <= lower_band and 
            z_score <= -self.parameters["entry_std_dev"] and
            rsi <= self.parameters["rsi_oversold"] and
            volume_confirmation):
            
            return Signal(
                symbol=self.price_history[-1].get("symbol", "UNKNOWN"),
                signal_type=Signal.BUY,
                timestamp=current_time,
                price=current_price,
                strength=min(1.0, abs(z_score) / 3),  # Normalize strength
                metadata={
                    "z_score": z_score,
                    "rsi": rsi,
                    "lower_band": lower_band,
                    "reason": "Mean reversion long entry"
                }
            )
        
        # Short entry conditions (price above upper band, overbought RSI)
        elif (current_price >= upper_band and 
              z_score >= self.parameters["entry_std_dev"] and
              rsi >= self.parameters["rsi_overbought"] and
              volume_confirmation):
            
            return Signal(
                symbol=self.price_history[-1].get("symbol", "UNKNOWN"),
                signal_type=Signal.SELL,
                timestamp=current_time,
                price=current_price,
                strength=min(1.0, abs(z_score) / 3),  # Normalize strength
                metadata={
                    "z_score": z_score,
                    "rsi": rsi,
                    "upper_band": upper_band,
                    "reason": "Mean reversion short entry"
                }
            )
        
        return None
    
    def _check_exit_signals(self, current_price: float, current_time: datetime, current_index: int) -> Optional[Signal]:
        """
        Check for exit signals.
        
        Args:
            current_price: Current price
            current_time: Current timestamp
            current_index: Current index in price history
            
        Returns:
            Exit signal or None
        """
        if not self.current_position or not self.position_entry_price:
            return None
        
        # Get latest indicator values
        z_score = self.indicators["z_score"][-1] if self.indicators["z_score"] else 0
        sma = self.indicators["sma"][-1] if self.indicators["sma"] else current_price
        atr = self.indicators["atr"][-1] if self.indicators["atr"] else (current_price * 0.01)  # Default to 1% of price
        
        # Variables for signal generation
        exit_reason = None
        
        # Check for long position exit
        if self.current_position == "LONG":
            # Stop loss calculation
            if self.parameters["use_atr_stops"]:
                stop_price = self.position_entry_price - (atr * self.parameters["atr_multiplier"])
            else:
                stop_price = self.position_entry_price * (1 - self.parameters["stop_loss_pct"])
            
            # Take profit
            take_profit = self.position_entry_price * (1 + self.parameters["take_profit_pct"])
            
            # Mean reversion exit (price returns to mean)
            mean_exit = abs(z_score) <= self.parameters["exit_std_dev"] or current_price >= sma
            
            # Maximum holding period
            max_holding_reached = (current_index - self.position_entry_index) >= self.parameters["max_holding_periods"]
            
            # Check exit conditions
            if current_price <= stop_price:
                exit_reason = "Stop loss"
            elif current_price >= take_profit:
                exit_reason = "Take profit"
            elif mean_exit:
                exit_reason = "Return to mean"
            elif max_holding_reached:
                exit_reason = "Max holding period reached"
        
        # Check for short position exit
        elif self.current_position == "SHORT":
            # Stop loss calculation
            if self.parameters["use_atr_stops"]:
                stop_price = self.position_entry_price + (atr * self.parameters["atr_multiplier"])
            else:
                stop_price = self.position_entry_price * (1 + self.parameters["stop_loss_pct"])
            
            # Take profit
            take_profit = self.position_entry_price * (1 - self.parameters["take_profit_pct"])
            
            # Mean reversion exit (price returns to mean)
            mean_exit = abs(z_score) <= self.parameters["exit_std_dev"] or current_price <= sma
            
            # Maximum holding period
            max_holding_reached = (current_index - self.position_entry_index) >= self.parameters["max_holding_periods"]
            
            # Check exit conditions
            if current_price >= stop_price:
                exit_reason = "Stop loss"
            elif current_price <= take_profit:
                exit_reason = "Take profit"
            elif mean_exit:
                exit_reason = "Return to mean"
            elif max_holding_reached:
                exit_reason = "Max holding period reached"
        
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
