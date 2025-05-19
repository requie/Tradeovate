"""
Implementation of machine learning based predictive trading strategy.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

from .strategy import Strategy, MarketData, Signal

class MLPredictiveStrategy(Strategy):
    """
    Machine Learning based predictive trading strategy.
    
    This strategy uses a Random Forest classifier to predict price movements
    based on technical indicators and generates trading signals accordingly.
    """
    
    def __init__(self, name: str = "MLPredictive", parameters: Dict[str, Any] = None):
        """
        Initialize ML-based strategy with parameters.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
        """
        default_params = {
            "lookback_window": 10,        # Number of periods to use for feature creation
            "prediction_horizon": 5,      # Number of periods ahead to predict
            "training_ratio": 0.7,        # Ratio of data to use for training
            "retrain_interval": 500,      # Number of bars between model retraining
            "min_training_samples": 100,  # Minimum samples required for training
            "confidence_threshold": 0.6,  # Minimum prediction confidence for signal generation
            "stop_loss_pct": 0.02,        # Stop loss percentage
            "take_profit_pct": 0.04,      # Take profit percentage
            "use_trailing_stop": True,    # Whether to use trailing stop
            "trailing_stop_pct": 0.015,   # Trailing stop percentage
            "model_path": "",             # Path to save/load model (empty for no persistence)
            "feature_importance_threshold": 0.02  # Minimum feature importance to keep feature
        }
        
        # Merge default parameters with provided parameters
        merged_params = default_params.copy()
        if parameters:
            merged_params.update(parameters)
            
        super().__init__(name, merged_params)
        
        # Initialize data storage
        self.price_history = []
        self.features = pd.DataFrame()
        self.model = None
        self.scaler = None
        self.last_train_index = 0
        self.current_position = None  # "LONG", "SHORT", or None
        self.position_entry_price = None
        self.position_entry_time = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
        self.feature_names = []
    
    def initialize(self) -> None:
        """Initialize strategy with parameters."""
        super().initialize()
        self.price_history = []
        
        # Try to load model if path is provided
        if self.parameters["model_path"] and os.path.exists(self.parameters["model_path"]):
            try:
                self.model = joblib.load(self.parameters["model_path"])
                scaler_path = self.parameters["model_path"].replace('.pkl', '_scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
                self.scaler = None
    
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
        
        # Update features if we have enough data
        if len(self.price_history) >= self.parameters["lookback_window"] + self.parameters["prediction_horizon"]:
            self._update_features()
            
            # Train or retrain model if needed
            current_index = len(self.price_history) - 1
            samples_since_last_train = current_index - self.last_train_index
            
            if (self.model is None or 
                (samples_since_last_train >= self.parameters["retrain_interval"] and 
                 len(self.features) >= self.parameters["min_training_samples"])):
                self._train_model()
                self.last_train_index = current_index
        
        # Update position tracking
        if self.current_position == "LONG":
            if data.high > (self.highest_price_since_entry or 0):
                self.highest_price_since_entry = data.high
        elif self.current_position == "SHORT":
            if data.low < (self.lowest_price_since_entry or float('inf')) or self.lowest_price_since_entry is None:
                self.lowest_price_since_entry = data.low
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for machine learning model.
        
        Args:
            df: DataFrame with price history
            
        Returns:
            DataFrame with features
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Basic price features
        lookback = self.parameters["lookback_window"]
        
        # Returns over different periods
        for period in [1, 2, 3, 5, 10]:
            if period < len(data):
                data[f'return_{period}'] = data['close'].pct_change(period)
        
        # Moving averages and their ratios
        for ma_period in [5, 10, 20, 50]:
            if ma_period < len(data):
                data[f'ma_{ma_period}'] = data['close'].rolling(window=ma_period).mean()
                data[f'close_to_ma_{ma_period}'] = data['close'] / data[f'ma_{ma_period}']
        
        # Volatility features
        for vol_period in [5, 10, 20]:
            if vol_period < len(data):
                data[f'volatility_{vol_period}'] = data['close'].rolling(window=vol_period).std() / data['close']
        
        # Volume features
        data['volume_change'] = data['volume'].pct_change()
        for vol_period in [5, 10]:
            if vol_period < len(data):
                data[f'volume_ma_{vol_period}'] = data['volume'].rolling(window=vol_period).mean()
                data[f'volume_ratio_{vol_period}'] = data['volume'] / data[f'volume_ma_{vol_period}']
        
        # Price range features
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # Technical indicators
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for bb_period in [20]:
            if bb_period < len(data):
                data[f'bb_middle_{bb_period}'] = data['close'].rolling(window=bb_period).mean()
                data[f'bb_std_{bb_period}'] = data['close'].rolling(window=bb_period).std()
                data[f'bb_upper_{bb_period}'] = data[f'bb_middle_{bb_period}'] + 2 * data[f'bb_std_{bb_period}']
                data[f'bb_lower_{bb_period}'] = data[f'bb_middle_{bb_period}'] - 2 * data[f'bb_std_{bb_period}']
                data[f'bb_width_{bb_period}'] = (data[f'bb_upper_{bb_period}'] - data[f'bb_lower_{bb_period}']) / data[f'bb_middle_{bb_period}']
                data[f'bb_position_{bb_period}'] = (data['close'] - data[f'bb_lower_{bb_period}']) / (data[f'bb_upper_{bb_period}'] - data[f'bb_lower_{bb_period}'])
        
        # MACD
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Drop NaN values
        data = data.dropna()
        
        # Store feature names for later use
        self.feature_names = [col for col in data.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return data
    
    def _update_features(self) -> None:
        """Update features based on price history."""
        # Convert price history to DataFrame
        df = pd.DataFrame(self.price_history)
        
        # Create features
        feature_df = self._create_features(df)
        
        # Create target variable (future returns)
        horizon = self.parameters["prediction_horizon"]
        if len(feature_df) > horizon:
            feature_df['future_return'] = feature_df['close'].pct_change(horizon).shift(-horizon)
            
            # Create binary target (1 for positive return, 0 for negative)
            feature_df['target'] = (feature_df['future_return'] > 0).astype(int)
        
        # Update features DataFrame
        self.features = feature_df
    
    def _train_model(self) -> None:
        """Train the machine learning model."""
        if len(self.features) < self.parameters["min_training_samples"]:
            return
        
        # Prepare training data
        features = self.features.dropna()
        
        if len(features) < self.parameters["min_training_samples"]:
            return
        
        # Split into features and target
        X = features[self.feature_names]
        y = features['target']
        
        # Split into training and validation sets
        train_size = int(len(features) * self.parameters["training_ratio"])
        X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if len(X_val) > 0 else None
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        if len(X_val) > 0:
            val_accuracy = self.model.score(X_val_scaled, y_val)
            print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Feature importance analysis
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Feature ranking:")
        for i, idx in enumerate(indices):
            if i < 10:  # Print top 10 features
                print(f"{i+1}. {self.feature_names[idx]} ({importances[idx]:.4f})")
        
        # Save model if path is provided
        if self.parameters["model_path"]:
            try:
                joblib.dump(self.model, self.parameters["model_path"])
                scaler_path = self.parameters["model_path"].replace('.pkl', '_scaler.pkl')
                joblib.dump(self.scaler, scaler_path)
            except Exception as e:
                print(f"Error saving model: {e}")
    
    def generate_signals(self) -> List[Signal]:
        """
        Generate trading signals based on processed data.
        
        Returns:
            List of trading signals
        """
        signals = []
        
        # Need model and enough data to generate signals
        if self.model is None or self.scaler is None or len(self.features) < 2:
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
        # Get latest feature values
        latest_features = self.features.iloc[-1:][self.feature_names]
        
        # Scale features
        scaled_features = self.scaler.transform(latest_features)
        
        # Make prediction
        prediction_proba = self.model.predict_proba(scaled_features)[0]
        prediction = self.model.predict(scaled_features)[0]
        
        # Get confidence
        confidence = prediction_proba[prediction]
        
        # Generate signal if confidence exceeds threshold
        if confidence >= self.parameters["confidence_threshold"]:
            if prediction == 1:  # Predicting price increase
                return Signal(
                    symbol=self.price_history[-1].get("symbol", "UNKNOWN"),
                    signal_type=Signal.BUY,
                    timestamp=current_time,
                    price=current_price,
                    strength=confidence,
                    metadata={
                        "prediction": "UP",
                        "confidence": confidence,
                        "reason": "ML prediction - price increase"
                    }
                )
            else:  # Predicting price decrease
                return Signal(
                    symbol=self.price_history[-1].get("symbol", "UNKNOWN"),
                    signal_type=Signal.SELL,
                    timestamp=current_time,
                    price=current_price,
                    strength=confidence,
                    metadata={
                        "prediction": "DOWN",
                        "confidence": confidence,
                        "reason": "ML prediction - price decrease"
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
