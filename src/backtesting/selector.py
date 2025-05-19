"""
Main module for strategy selection and backtesting integration.
"""
from typing import Dict, List, Any, Optional, Type
import pandas as pd
from datetime import datetime
import logging

from .strategy import Strategy
from .engine import BacktestEngine, StrategyEvaluator
from .momentum_strategy import MomentumStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .breakout_strategy import BreakoutStrategy
from .ml_predictive_strategy import MLPredictiveStrategy

logger = logging.getLogger(__name__)

class StrategySelector:
    """
    Selects the best trading strategy based on backtesting results.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.0,
                 slippage: float = 0.0):
        """
        Initialize the strategy selector.
        
        Args:
            initial_capital: Initial capital for backtesting
            commission_rate: Commission rate per trade (percentage)
            slippage: Slippage per trade (percentage)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.evaluator = StrategyEvaluator(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage=slippage
        )
        
        # Available strategy classes
        self.strategy_classes = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy,
            "breakout": BreakoutStrategy,
            "ml_predictive": MLPredictiveStrategy
        }
    
    def create_strategy(self, 
                       strategy_type: str, 
                       name: Optional[str] = None,
                       parameters: Optional[Dict[str, Any]] = None) -> Strategy:
        """
        Create a strategy instance of the specified type.
        
        Args:
            strategy_type: Type of strategy to create
            name: Optional name for the strategy
            parameters: Optional parameters for the strategy
            
        Returns:
            Strategy instance
        """
        if strategy_type not in self.strategy_classes:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy_class = self.strategy_classes[strategy_type]
        
        # Use default name if not provided
        if name is None:
            name = strategy_type.capitalize()
        
        return strategy_class(name=name, parameters=parameters)
    
    def select_best_strategy(self, 
                            data: pd.DataFrame,
                            strategy_configs: List[Dict[str, Any]],
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            selection_metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Select the best strategy based on backtesting results.
        
        Args:
            data: Historical data for backtesting
            strategy_configs: List of strategy configurations
            start_date: Start date for backtesting
            end_date: End date for backtesting
            selection_metric: Metric to use for selection
            
        Returns:
            Dictionary with best strategy information
        """
        # Create strategies from configurations
        strategies = []
        
        for config in strategy_configs:
            strategy_type = config.get("type")
            name = config.get("name")
            parameters = config.get("parameters", {})
            
            strategy = self.create_strategy(
                strategy_type=strategy_type,
                name=name or strategy_type,
                parameters=parameters
            )
            
            strategies.append(strategy)
        
        # Evaluate all strategies
        results = self.evaluator.evaluate_strategies(
            strategies=strategies,
            data=data,
            start_date=start_date,
            end_date=end_date
        )
        
        # Create comparison table
        comparison = self.evaluator.compare_strategies(results)
        
        # Select best strategy
        best_strategy_name, best_result = self.evaluator.select_best_strategy(
            results=results,
            metric=selection_metric
        )
        
        # Find the configuration for the best strategy
        best_config = next((config for config in strategy_configs 
                           if config.get("name") == best_strategy_name or 
                           (config.get("type") == best_strategy_name.lower() and not config.get("name"))),
                          None)
        
        return {
            "best_strategy_name": best_strategy_name,
            "best_strategy_type": best_config.get("type") if best_config else None,
            "best_strategy_parameters": best_config.get("parameters", {}) if best_config else None,
            "best_strategy_metrics": best_result.metrics,
            "comparison_table": comparison,
            "all_results": results
        }
