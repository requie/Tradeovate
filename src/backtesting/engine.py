"""
Backtesting engine for evaluating trading strategies.
"""
from typing import Dict, List, Any, Optional, Tuple, Type
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from .strategy import Strategy, MarketData, Signal

logger = logging.getLogger(__name__)

class Trade:
    """Represents a trade executed during backtesting."""
    
    def __init__(self, 
                 symbol: str,
                 entry_time: datetime,
                 entry_price: float,
                 direction: str,  # "LONG" or "SHORT"
                 quantity: float,
                 exit_time: Optional[datetime] = None,
                 exit_price: Optional[float] = None,
                 pnl: Optional[float] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a trade.
        
        Args:
            symbol: Trading symbol
            entry_time: Entry timestamp
            entry_price: Entry price
            direction: Trade direction ("LONG" or "SHORT")
            quantity: Trade size
            exit_time: Exit timestamp (None if open)
            exit_price: Exit price (None if open)
            pnl: Profit/loss (None if open)
            metadata: Additional trade information
        """
        self.symbol = symbol
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.quantity = quantity
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.pnl = pnl
        self.metadata = metadata or {}
        self.is_open = exit_time is None
    
    def close(self, exit_time: datetime, exit_price: float) -> None:
        """
        Close an open trade.
        
        Args:
            exit_time: Exit timestamp
            exit_price: Exit price
        """
        self.exit_time = exit_time
        self.exit_price = exit_price
        
        if self.direction == "LONG":
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.pnl = (self.entry_price - exit_price) * self.quantity
            
        self.is_open = False
    
    def current_pnl(self, current_price: float) -> float:
        """
        Calculate current unrealized P&L for an open trade.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized P&L
        """
        if not self.is_open:
            return self.pnl or 0.0
            
        if self.direction == "LONG":
            return (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - current_price) * self.quantity
    
    def duration(self) -> Optional[timedelta]:
        """
        Calculate trade duration.
        
        Returns:
            Trade duration or None if trade is still open
        """
        if self.is_open:
            return None
        return self.exit_time - self.entry_time
    
    def __str__(self) -> str:
        status = "OPEN" if self.is_open else "CLOSED"
        pnl_str = f"PnL: {self.pnl:.2f}" if self.pnl is not None else "PnL: N/A"
        return f"Trade({self.symbol}, {self.direction}, {status}, Entry: {self.entry_price}, Exit: {self.exit_price}, {pnl_str})"

class BacktestResult:
    """Container for backtest results."""
    
    def __init__(self, 
                 strategy_name: str,
                 trades: List[Trade],
                 equity_curve: pd.Series,
                 parameters: Dict[str, Any],
                 start_date: datetime,
                 end_date: datetime,
                 initial_capital: float):
        """
        Initialize backtest results.
        
        Args:
            strategy_name: Name of the strategy
            trades: List of executed trades
            equity_curve: Equity curve as pandas Series
            parameters: Strategy parameters used
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital
        """
        self.strategy_name = strategy_name
        self.trades = trades
        self.equity_curve = equity_curve
        self.parameters = parameters
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # Calculate performance metrics
        self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Skip metrics calculation if no trades
        if not self.trades:
            logger.warning(f"No trades executed for strategy {self.strategy_name}")
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "annualized_return": 0.0
            }
        
        # Basic trade metrics
        closed_trades = [t for t in self.trades if not t.is_open]
        metrics["total_trades"] = len(closed_trades)
        
        if closed_trades:
            winning_trades = [t for t in closed_trades if t.pnl > 0]
            losing_trades = [t for t in closed_trades if t.pnl <= 0]
            
            metrics["win_rate"] = len(winning_trades) / len(closed_trades) if closed_trades else 0.0
            
            total_profit = sum(t.pnl for t in winning_trades)
            total_loss = abs(sum(t.pnl for t in losing_trades))
            
            metrics["profit_factor"] = total_profit / total_loss if total_loss > 0 else float('inf')
            metrics["average_profit"] = total_profit / len(winning_trades) if winning_trades else 0.0
            metrics["average_loss"] = total_loss / len(losing_trades) if losing_trades else 0.0
            metrics["total_pnl"] = sum(t.pnl for t in closed_trades)
        
        # Equity curve metrics
        if not self.equity_curve.empty:
            # Calculate returns
            returns = self.equity_curve.pct_change().dropna()
            
            # Sharpe Ratio (annualized, assuming daily data)
            if len(returns) > 1:
                sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
                metrics["sharpe_ratio"] = sharpe
            else:
                metrics["sharpe_ratio"] = 0.0
            
            # Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative / running_max - 1)
            metrics["max_drawdown"] = abs(drawdown.min())
            
            # Total and annualized return
            total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
            metrics["total_return"] = total_return
            
            # Calculate annualized return
            days = (self.end_date - self.start_date).days
            if days > 0:
                metrics["annualized_return"] = (1 + total_return) ** (365 / days) - 1
            else:
                metrics["annualized_return"] = 0.0
        
        return metrics
    
    def summary(self) -> str:
        """
        Generate a summary of backtest results.
        
        Returns:
            Summary string
        """
        summary = [
            f"Strategy: {self.strategy_name}",
            f"Period: {self.start_date.date()} to {self.end_date.date()}",
            f"Initial Capital: ${self.initial_capital:.2f}",
            f"Final Capital: ${self.equity_curve.iloc[-1]:.2f}",
            f"Total Return: {self.metrics.get('total_return', 0.0)*100:.2f}%",
            f"Annualized Return: {self.metrics.get('annualized_return', 0.0)*100:.2f}%",
            f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0.0):.2f}",
            f"Max Drawdown: {self.metrics.get('max_drawdown', 0.0)*100:.2f}%",
            f"Profit Factor: {self.metrics.get('profit_factor', 0.0):.2f}",
            f"Win Rate: {self.metrics.get('win_rate', 0.0)*100:.2f}%",
            f"Total Trades: {self.metrics.get('total_trades', 0)}"
        ]
        
        return "\n".join(summary)

class BacktestEngine:
    """Engine for backtesting trading strategies."""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.0,
                 slippage: float = 0.0):
        """
        Initialize the backtest engine.
        
        Args:
            initial_capital: Initial capital for backtesting
            commission_rate: Commission rate per trade (percentage)
            slippage: Slippage per trade (percentage)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.reset()
    
    def reset(self) -> None:
        """Reset the backtest engine state."""
        self.current_capital = self.initial_capital
        self.equity_history = []
        self.trades = []
        self.open_trades = []
        self.current_time = None
    
    def run_backtest(self, 
                    strategy: Strategy, 
                    data: pd.DataFrame,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> BacktestResult:
        """
        Run a backtest for a strategy on historical data.
        
        Args:
            strategy: Strategy to backtest
            data: Historical data as pandas DataFrame
            start_date: Start date for backtest (None for all data)
            end_date: End date for backtest (None for all data)
            
        Returns:
            BacktestResult object with performance metrics
        """
        self.reset()
        
        # Initialize strategy
        if not strategy.is_initialized:
            strategy.initialize()
        
        # Filter data by date range if specified
        if start_date or end_date:
            mask = pd.Series(True, index=data.index)
            if start_date:
                mask &= data.index >= start_date
            if end_date:
                mask &= data.index <= end_date
            data = data[mask]
        
        if data.empty:
            logger.warning("No data available for backtesting")
            return self._create_empty_result(strategy, start_date, end_date)
        
        # Ensure data is sorted by date
        data = data.sort_index()
        
        # Track equity curve
        equity_curve = pd.Series(index=data.index, dtype=float)
        
        # Process each data point
        for timestamp, row in data.iterrows():
            self.current_time = timestamp
            
            # Create market data object
            market_data = MarketData(
                symbol=row.get('symbol', 'UNKNOWN'),
                timestamp=timestamp,
                open_price=row.get('open', row.get('Open', 0.0)),
                high_price=row.get('high', row.get('High', 0.0)),
                low_price=row.get('low', row.get('Low', 0.0)),
                close_price=row.get('close', row.get('Close', 0.0)),
                volume=row.get('volume', row.get('Volume', 0.0))
            )
            
            # Process market data with strategy
            strategy.process_market_data(market_data)
            
            # Generate signals
            signals = strategy.generate_signals()
            
            # Execute signals
            for signal in signals:
                self._process_signal(signal, market_data)
            
            # Update open trade P&L
            self._update_open_trades(market_data.close)
            
            # Record equity
            current_equity = self.current_capital + sum(t.current_pnl(market_data.close) for t in self.open_trades)
            equity_curve[timestamp] = current_equity
        
        # Close any remaining open trades at the last price
        if self.open_trades and not data.empty:
            last_price = data.iloc[-1].get('close', data.iloc[-1].get('Close', 0.0))
            last_time = data.index[-1]
            
            for trade in self.open_trades[:]:
                self._close_trade(trade, last_time, last_price)
        
        # Create backtest result
        actual_start = data.index[0] if not data.empty else start_date or datetime.now()
        actual_end = data.index[-1] if not data.empty else end_date or datetime.now()
        
        result = BacktestResult(
            strategy_name=strategy.name,
            trades=self.trades,
            equity_curve=equity_curve,
            parameters=strategy.parameters,
            start_date=actual_start,
            end_date=actual_end,
            initial_capital=self.initial_capital
        )
        
        return result
    
    def _create_empty_result(self, 
                            strategy: Strategy, 
                            start_date: Optional[datetime], 
                            end_date: Optional[datetime]) -> BacktestResult:
        """
        Create an empty result when no data is available.
        
        Args:
            strategy: Strategy that was tested
            start_date: Requested start date
            end_date: Requested end date
            
        Returns:
            Empty BacktestResult
        """
        empty_equity = pd.Series([self.initial_capital], index=[datetime.now()])
        
        return BacktestResult(
            strategy_name=strategy.name,
            trades=[],
            equity_curve=empty_equity,
            parameters=strategy.parameters,
            start_date=start_date or datetime.now(),
            end_date=end_date or datetime.now(),
            initial_capital=self.initial_capital
        )
    
    def _process_signal(self, signal: Signal, market_data: MarketData) -> None:
        """
        Process a trading signal.
        
        Args:
            signal: Trading signal to process
            market_data: Current market data
        """
        if signal.signal_type == Signal.BUY:
            self._open_long_trade(signal, market_data)
        elif signal.signal_type == Signal.SELL:
            self._open_short_trade(signal, market_data)
        elif signal.signal_type == Signal.CLOSE_LONG:
            self._close_long_trades(signal, market_data)
        elif signal.signal_type == Signal.CLOSE_SHORT:
            self._close_short_trades(signal, market_data)
    
    def _open_long_trade(self, signal: Signal, market_data: MarketData) -> None:
        """
        Open a long trade.
        
        Args:
            signal: Buy signal
            market_data: Current market data
        """
        # Calculate position size (simplified)
        position_size = self._calculate_position_size(signal, market_data)
        
        if position_size <= 0:
            return
        
        # Apply slippage to entry price
        entry_price = signal.price * (1 + self.slippage)
        
        # Create and record trade
        trade = Trade(
            symbol=signal.symbol,
            entry_time=signal.timestamp,
            entry_price=entry_price,
            direction="LONG",
            quantity=position_size,
            metadata={"signal": signal.metadata}
        )
        
        self.trades.append(trade)
        self.open_trades.append(trade)
    
    def _open_short_trade(self, signal: Signal, market_data: MarketData) -> None:
        """
        Open a short trade.
        
        Args:
            signal: Sell signal
            market_data: Current market data
        """
        # Calculate position size (simplified)
        position_size = self._calculate_position_size(signal, market_data)
        
        if position_size <= 0:
            return
        
        # Apply slippage to entry price
        entry_price = signal.price * (1 - self.slippage)
        
        # Create and record trade
        trade = Trade(
            symbol=signal.symbol,
            entry_time=signal.timestamp,
            entry_price=entry_price,
            direction="SHORT",
            quantity=position_size,
            metadata={"signal": signal.metadata}
        )
        
        self.trades.append(trade)
        self.open_trades.append(trade)
    
    def _close_long_trades(self, signal: Signal, market_data: MarketData) -> None:
        """
        Close open long trades.
        
        Args:
            signal: Close signal
            market_data: Current market data
        """
        # Apply slippage to exit price
        exit_price = signal.price * (1 - self.slippage)
        
        # Find open long trades for this symbol
        for trade in self.open_trades[:]:
            if trade.symbol == signal.symbol and trade.direction == "LONG":
                self._close_trade(trade, signal.timestamp, exit_price)
    
    def _close_short_trades(self, signal: Signal, market_data: MarketData) -> None:
        """
        Close open short trades.
        
        Args:
            signal: Close signal
            market_data: Current market data
        """
        # Apply slippage to exit price
        exit_price = signal.price * (1 + self.slippage)
        
        # Find open short trades for this symbol
        for trade in self.open_trades[:]:
            if trade.symbol == signal.symbol and trade.direction == "SHORT":
                self._close_trade(trade, signal.timestamp, exit_price)
    
    def _close_trade(self, trade: Trade, exit_time: datetime, exit_price: float) -> None:
        """
        Close a specific trade.
        
        Args:
            trade: Trade to close
            exit_time: Exit timestamp
            exit_price: Exit price
        """
        if trade in self.open_trades:
            # Close the trade
            trade.close(exit_time, exit_price)
            
            # Apply commission
            commission = trade.quantity * exit_price * self.commission_rate
            trade.pnl -= commission
            
            # Update capital
            self.current_capital += trade.pnl
            
            # Remove from open trades
            self.open_trades.remove(trade)
    
    def _update_open_trades(self, current_price: float) -> None:
        """
        Update unrealized P&L for open trades.
        
        Args:
            current_price: Current market price
        """
        # This is just for tracking purposes, no actual changes to trades
        pass
    
    def _calculate_position_size(self, signal: Signal, market_data: MarketData) -> float:
        """
        Calculate position size based on available capital and risk parameters.
        
        Args:
            signal: Trading signal
            market_data: Current market data
            
        Returns:
            Position size
        """
        # Simple position sizing - use 10% of available capital
        # In a real implementation, this would use more sophisticated risk management
        available_capital = self.current_capital * 0.1
        
        # Calculate quantity based on price
        quantity = available_capital / signal.price
        
        return quantity

class StrategyEvaluator:
    """Evaluates and compares multiple trading strategies."""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.0,
                 slippage: float = 0.0):
        """
        Initialize the strategy evaluator.
        
        Args:
            initial_capital: Initial capital for backtesting
            commission_rate: Commission rate per trade (percentage)
            slippage: Slippage per trade (percentage)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.backtest_engine = BacktestEngine(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage=slippage
        )
    
    def evaluate_strategies(self, 
                           strategies: List[Strategy], 
                           data: pd.DataFrame,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> Dict[str, BacktestResult]:
        """
        Evaluate multiple strategies on the same dataset.
        
        Args:
            strategies: List of strategies to evaluate
            data: Historical data as pandas DataFrame
            start_date: Start date for evaluation
            end_date: End date for evaluation
            
        Returns:
            Dictionary mapping strategy names to backtest results
        """
        results = {}
        
        for strategy in strategies:
            logger.info(f"Evaluating strategy: {strategy.name}")
            result = self.backtest_engine.run_backtest(
                strategy=strategy,
                data=data,
                start_date=start_date,
                end_date=end_date
            )
            results[strategy.name] = result
            logger.info(f"Completed evaluation of {strategy.name}")
        
        return results
    
    def select_best_strategy(self, 
                            results: Dict[str, BacktestResult],
                            metric: str = "sharpe_ratio") -> Tuple[str, BacktestResult]:
        """
        Select the best strategy based on a performance metric.
        
        Args:
            results: Dictionary of backtest results
            metric: Metric to use for selection
            
        Returns:
            Tuple of (best strategy name, best strategy result)
        """
        if not results:
            raise ValueError("No results provided for strategy selection")
        
        # Get metric values for each strategy
        metric_values = {name: result.metrics.get(metric, float('-inf')) 
                        for name, result in results.items()}
        
        # Find best strategy
        best_strategy = max(metric_values.items(), key=lambda x: x[1])[0]
        
        return best_strategy, results[best_strategy]
    
    def compare_strategies(self, results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """
        Compare multiple strategies across various metrics.
        
        Args:
            results: Dictionary of backtest results
            
        Returns:
            DataFrame with comparison of strategies
        """
        if not results:
            return pd.DataFrame()
        
        # Metrics to compare
        metrics = [
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "profit_factor",
            "win_rate",
            "total_trades"
        ]
        
        # Create comparison DataFrame
        comparison = {}
        
        for name, result in results.items():
            comparison[name] = {metric: result.metrics.get(metric, 0.0) for metric in metrics}
        
        return pd.DataFrame(comparison).T
