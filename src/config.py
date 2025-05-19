"""
Configuration module for the trading bot.
"""
from typing import Dict, Any, List, Optional, Union
import os
import json
import yaml
import logging
import copy
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages configuration for the trading bot.
    
    This class handles loading, validating, and saving configuration
    from JSON or YAML files.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        "mode": "simulation",  # "live" or "simulation"
        "account_id": None,
        "initial_capital": 100000.0,
        "commission_rate": 0.0,
        "slippage": 0.0,
        "status_interval_minutes": 15,
        "simulation_speed": 1.0,
        "simulation_data_dir": "data",
        "api": {
            "username": "",
            "password": "",
            "app_id": "",
            "app_version": "1.0",
            "cid": 0,
            "sec": "",
            "device_id": None,
            "use_live": False,
            "token_refresh_margin": 300,
            "max_retries": 3,
            "retry_delay": 2
        },
        "risk": {
            "max_position_size": 1,
            "max_positions": 3,
            "max_drawdown_pct": 5.0,
            "max_daily_loss_pct": 2.0,
            "max_trade_risk_pct": 1.0,
            "position_size_method": "fixed",  # "fixed", "percent_risk", "kelly"
            "kelly_fraction": 0.5,
            "use_stop_loss": True,
            "stop_loss_atr_multiple": 2.0,
            "fixed_stop_loss_pct": 1.0,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiple": 1.5,
            "fixed_trailing_stop_pct": 0.75,
            "use_take_profit": True,
            "take_profit_atr_multiple": 3.0,
            "fixed_take_profit_pct": 2.0,
            "max_open_time_minutes": 240
        },
        "strategies": []
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            Exception: If configuration file cannot be loaded
        """
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found: {config_path}")
            return self.config
        
        try:
            # Determine file format
            if config_path.endswith(".json"):
                with open(config_path, "r") as f:
                    loaded_config = json.load(f)
            elif config_path.endswith((".yaml", ".yml")):
                with open(config_path, "r") as f:
                    loaded_config = yaml.safe_load(f)
            else:
                raise Exception(f"Unsupported configuration file format: {config_path}")
            
            # Update configuration
            self._update_config(loaded_config)
            
            # Store config path
            self.config_path = config_path
            
            logger.info(f"Loaded configuration from {config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def save(self, config_path: Optional[str] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration file (uses current path if None)
            
        Returns:
            True if configuration was saved successfully
        """
        # Use current path if not specified
        if not config_path:
            config_path = self.config_path
        
        if not config_path:
            logger.error("No configuration path specified")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
            
            # Determine file format
            if config_path.endswith(".json"):
                with open(config_path, "w") as f:
                    json.dump(self.config, f, indent=2)
            elif config_path.endswith((".yaml", ".yml")):
                with open(config_path, "w") as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                raise Exception(f"Unsupported configuration file format: {config_path}")
            
            logger.info(f"Saved configuration to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def add_strategy(self, 
                    strategy_type: str,
                    symbol: str,
                    name: Optional[str] = None,
                    parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a strategy configuration.
        
        Args:
            strategy_type: Strategy type
            symbol: Trading symbol
            name: Strategy name (optional)
            parameters: Strategy parameters (optional)
        """
        # Create strategy configuration
        strategy_config = {
            "type": strategy_type,
            "symbol": symbol
        }
        
        if name:
            strategy_config["name"] = name
        
        if parameters:
            strategy_config["parameters"] = parameters
        
        # Add to strategies list
        if "strategies" not in self.config:
            self.config["strategies"] = []
        
        self.config["strategies"].append(strategy_config)
    
    def remove_strategy(self, index: int) -> bool:
        """
        Remove a strategy configuration.
        
        Args:
            index: Strategy index
            
        Returns:
            True if strategy was removed
        """
        if "strategies" not in self.config or index >= len(self.config["strategies"]):
            return False
        
        del self.config["strategies"][index]
        return True
    
    def validate(self) -> List[str]:
        """
        Validate the configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        if self.config.get("mode") not in ["live", "simulation"]:
            errors.append("Invalid mode: must be 'live' or 'simulation'")
        
        # Check API configuration for live mode
        if self.config.get("mode") == "live":
            api_config = self.config.get("api", {})
            
            if not api_config.get("username"):
                errors.append("Missing API username")
            
            if not api_config.get("password"):
                errors.append("Missing API password")
            
            if not api_config.get("app_id"):
                errors.append("Missing API app_id")
            
            if not api_config.get("cid"):
                errors.append("Missing API cid")
            
            if not api_config.get("sec"):
                errors.append("Missing API sec")
            
            if not self.config.get("account_id"):
                errors.append("Missing account_id for live mode")
        
        # Check strategies
        strategies = self.config.get("strategies", [])
        if not strategies:
            errors.append("No strategies configured")
        
        for i, strategy in enumerate(strategies):
            if not strategy.get("type"):
                errors.append(f"Strategy {i}: Missing type")
            
            if not strategy.get("symbol"):
                errors.append(f"Strategy {i}: Missing symbol")
        
        return errors
    
    def create_default_config(self, config_path: str) -> bool:
        """
        Create a default configuration file.
        
        Args:
            config_path: Path to save configuration file
            
        Returns:
            True if configuration was created successfully
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
            
            # Reset to default configuration
            self.config = copy.deepcopy(self.DEFAULT_CONFIG)
            
            # Add example strategy
            self.add_strategy(
                strategy_type="momentum",
                symbol="ES",
                name="ES Momentum",
                parameters={
                    "fast_period": 10,
                    "slow_period": 30,
                    "signal_threshold": 0.0005,
                    "stop_loss_pct": 0.5,
                    "take_profit_pct": 1.0,
                    "use_trailing_stop": True,
                    "trailing_stop_pct": 0.3
                }
            )
            
            # Save configuration
            return self.save(config_path)
            
        except Exception as e:
            logger.error(f"Error creating default configuration: {str(e)}")
            return False
    
    def _update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            new_config: New configuration values
        """
        # Deep merge configuration
        self._deep_update(self.config, new_config)
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep update a dictionary.
        
        Args:
            target: Target dictionary
            source: Source dictionary
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
