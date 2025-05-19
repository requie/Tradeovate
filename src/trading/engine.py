"""
Trading engine for real-time trading with Tradeovate API.
"""
from typing import Dict, Any, List, Optional, Union, Callable
import logging
import threading
import queue
import time
from datetime import datetime, timedelta
import json
import os
import yaml

from ..api.client import TradeovateApiClient, TradeovateApiError
from ..api.account import AccountManager
from ..api.order import OrderManager, Order
from ..api.market_data import MarketData
from ..backtesting.strategy import Strategy, MarketData as StrategyMarketData, Signal
from ..backtesting.selector import StrategySelector
from .risk_manager import RiskManager, RiskParameters

logger = logging.getLogger(__name__)

class TradingEngine:
    """
    Main trading engine for the Tradeovate trading bot.
    
    This class coordinates all components of the trading system,
    including API clients, market data, strategy execution,
    order management, and risk management.
    """
    
    # Trading modes
    MODE_LIVE = "live"
    MODE_SIMULATION = "simulation"
    
    def __init__(self, config_path: str):
        """
        Initialize the trading engine.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Trading mode
        self.mode = self.config.get("mode", self.MODE_SIMULATION)
        
        # API client
        self.api_client = self._create_api_client()
        
        # Core components
        self.account_manager = AccountManager(self.api_client)
        self.order_manager = OrderManager(self.api_client)
        self.market_data = MarketData(self.api_client)
        
        # Risk management
        risk_params = self._create_risk_parameters()
        self.risk_manager = RiskManager(
            account_manager=self.account_manager,
            order_manager=self.order_manager,
            market_data=self.market_data,
            risk_parameters=risk_params
        )
        
        # Strategy management
        self.strategy_selector = StrategySelector(
            initial_capital=self.config.get("initial_capital", 100000.0),
            commission_rate=self.config.get("commission_rate", 0.0),
            slippage=self.config.get("slippage", 0.0)
        )
        
        # Active strategies
        self.strategies = {}  # symbol -> Strategy
        self.active_symbols = set()
        
        # Engine state
        self.is_running = False
        self.main_thread = None
        self.stop_event = threading.Event()
        
        # For simulation mode
        self.simulation_data = {}  # symbol -> DataFrame
        self.simulation_index = {}  # symbol -> current index
        self.simulation_speed = self.config.get("simulation_speed", 1.0)  # Speed multiplier
        
        # Performance tracking
        self.start_time = None
        self.last_status_time = None
        self.status_interval = timedelta(minutes=self.config.get("status_interval_minutes", 15))
        
        logger.info(f"Trading engine initialized in {self.mode} mode")
    
    def start(self) -> None:
        """Start the trading engine."""
        if self.is_running:
            logger.warning("Trading engine is already running")
            return
        
        try:
            # Authenticate API client
            if self.mode == self.MODE_LIVE:
                self.api_client.authenticate()
            
            # Start core components
            if self.mode == self.MODE_LIVE:
                self.order_manager.start()
                self.market_data.start()
            
            # Initialize strategies
            self._initialize_strategies()
            
            # Set engine state
            self.is_running = True
            self.stop_event.clear()
            self.start_time = datetime.now()
            self.last_status_time = self.start_time
            
            # Start main thread
            self.main_thread = threading.Thread(target=self._main_loop)
            self.main_thread.daemon = True
            self.main_thread.start()
            
            logger.info(f"Trading engine started in {self.mode} mode")
            
        except Exception as e:
            logger.error(f"Error starting trading engine: {str(e)}")
            self.stop()
    
    def stop(self) -> None:
        """Stop the trading engine."""
        if not self.is_running:
            logger.warning("Trading engine is not running")
            return
        
        # Set stop event
        self.stop_event.set()
        
        # Wait for main thread to exit
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=30.0)
        
        # Stop core components
        if self.mode == self.MODE_LIVE:
            self.market_data.stop()
            self.order_manager.stop()
        
        # Close API client
        if self.mode == self.MODE_LIVE:
            self.api_client.close()
        
        # Set engine state
        self.is_running = False
        
        logger.info("Trading engine stopped")
    
    def add_symbol(self, symbol: str) -> bool:
        """
        Add a symbol to trade.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if symbol was added successfully
        """
        if symbol in self.active_symbols:
            logger.warning(f"Symbol {symbol} is already active")
            return False
        
        try:
            # Find contract
            contract = self.account_manager.find_contract(symbol)
            if not contract:
                logger.error(f"Contract not found for symbol {symbol}")
                return False
            
            contract_id = contract.get("id")
            if not contract_id:
                logger.error(f"Invalid contract for symbol {symbol}")
                return False
            
            # Subscribe to market data
            if self.mode == self.MODE_LIVE:
                self.market_data.subscribe(symbol)
            else:
                # Load historical data for simulation
                self._load_simulation_data(symbol)
            
            # Add to active symbols
            self.active_symbols.add(symbol)
            
            logger.info(f"Added symbol {symbol} (contract ID: {contract_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {str(e)}")
            return False
    
    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol from trading.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if symbol was removed successfully
        """
        if symbol not in self.active_symbols:
            logger.warning(f"Symbol {symbol} is not active")
            return False
        
        try:
            # Unsubscribe from market data
            if self.mode == self.MODE_LIVE:
                self.market_data.unsubscribe(symbol)
            
            # Remove strategy if exists
            if symbol in self.strategies:
                del self.strategies[symbol]
            
            # Remove from active symbols
            self.active_symbols.remove(symbol)
            
            logger.info(f"Removed symbol {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing symbol {symbol}: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the trading engine.
        
        Returns:
            Status information
        """
        # Get account information
        account_id = self.config.get("account_id")
        account = None
        positions = []
        
        if account_id and self.mode == self.MODE_LIVE:
            account = self.account_manager.get_account(account_id)
            positions = self.account_manager.get_positions(account_id)
        
        # Calculate runtime
        runtime = datetime.now() - self.start_time if self.start_time else timedelta()
        
        # Build status
        status = {
            "mode": self.mode,
            "running": self.is_running,
            "runtime": str(runtime),
            "active_symbols": list(self.active_symbols),
            "account": {
                "id": account_id,
                "balance": account.cash_balance if account else None,
                "pnl": account.pnl if account else None,
                "pnl_percentage": account.pnl_percentage if account else None
            },
            "positions": [
                {
                    "symbol": self._get_symbol_for_contract(position.contract_id),
                    "direction": "LONG" if position.is_long else "SHORT",
                    "size": position.size,
                    "entry_price": position.net_price,
                    "pnl": position.total_pnl
                }
                for position in positions
            ],
            "strategies": {
                symbol: strategy.__class__.__name__
                for symbol, strategy in self.strategies.items()
            }
        }
        
        return status
    
    def _main_loop(self) -> None:
        """Main trading loop."""
        logger.info("Starting main trading loop")
        
        while not self.stop_event.is_set():
            try:
                # Process market data
                if self.mode == self.MODE_LIVE:
                    self._process_live_data()
                else:
                    self._process_simulation_data()
                
                # Check risk limits
                if self.mode == self.MODE_LIVE:
                    account_id = self.config.get("account_id")
                    if account_id:
                        if not self.risk_manager.check_risk_limits(account_id):
                            logger.warning("Risk limits exceeded, closing all positions")
                            self.risk_manager.close_all_positions(account_id)
                
                # Log status periodically
                now = datetime.now()
                if self.last_status_time and now - self.last_status_time >= self.status_interval:
                    self._log_status()
                    self.last_status_time = now
                
                # Sleep to avoid excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(1.0)  # Sleep longer on error
    
    def _process_live_data(self) -> None:
        """Process live market data."""
        for symbol in self.active_symbols:
            try:
                # Get latest data
                data = self.market_data.get_historical_data(symbol, limit=100)
                if data.empty:
                    continue
                
                # Get latest price
                latest_price = self.market_data.get_latest_price(symbol)
                if not latest_price:
                    continue
                
                # Convert to strategy market data
                market_data = self._convert_to_strategy_market_data(symbol, data.iloc[-1])
                
                # Process with strategy
                if symbol in self.strategies:
                    strategy = self.strategies[symbol]
                    
                    # Process market data
                    strategy.process_market_data(market_data)
                    
                    # Generate signals
                    signals = strategy.generate_signals()
                    
                    # Process signals
                    if signals:
                        self._process_signals(symbol, signals)
                
                # Manage existing positions
                account_id = self.config.get("account_id")
                if account_id:
                    # Find contract ID for symbol
                    contract = self.account_manager.find_contract(symbol)
                    if contract:
                        contract_id = contract.get("id")
                        if contract_id:
                            # Manage position risk
                            self.risk_manager.manage_position_risk(
                                account_id=account_id,
                                contract_id=contract_id,
                                current_price=latest_price
                            )
                
            except Exception as e:
                logger.error(f"Error processing live data for {symbol}: {str(e)}")
    
    def _process_simulation_data(self) -> None:
        """Process simulation market data."""
        for symbol in self.active_symbols:
            try:
                # Check if we have simulation data
                if symbol not in self.simulation_data or symbol not in self.simulation_index:
                    continue
                
                data = self.simulation_data[symbol]
                index = self.simulation_index[symbol]
                
                # Check if we've reached the end of data
                if index >= len(data):
                    logger.info(f"End of simulation data for {symbol}")
                    self.remove_symbol(symbol)
                    continue
                
                # Get current bar
                current_bar = data.iloc[index]
                
                # Convert to strategy market data
                market_data = self._convert_to_strategy_market_data(symbol, current_bar)
                
                # Process with strategy
                if symbol in self.strategies:
                    strategy = self.strategies[symbol]
                    
                    # Process market data
                    strategy.process_market_data(market_data)
                    
                    # Generate signals
                    signals = strategy.generate_signals()
                    
                    # Process signals
                    if signals:
                        self._process_simulation_signals(symbol, signals, current_bar)
                
                # Increment index
                self.simulation_index[symbol] += 1
                
                # Sleep based on simulation speed
                if self.simulation_speed > 0:
                    time.sleep(1.0 / self.simulation_speed)
                
            except Exception as e:
                logger.error(f"Error processing simulation data for {symbol}: {str(e)}")
    
    def _process_signals(self, symbol: str, signals: List[Signal]) -> None:
        """
        Process trading signals.
        
        Args:
            symbol: Trading symbol
            signals: List of signals
        """
        account_id = self.config.get("account_id")
        if not account_id:
            logger.error("Cannot process signals: No account ID configured")
            return
        
        # Find contract
        contract = self.account_manager.find_contract(symbol)
        if not contract:
            logger.error(f"Contract not found for symbol {symbol}")
            return
        
        contract_id = contract.get("id")
        if not contract_id:
            logger.error(f"Invalid contract for symbol {symbol}")
            return
        
        # Get current position
        position = self.account_manager.get_position(account_id, contract_id)
        
        for signal in signals:
            try:
                # Process based on signal type
                if signal.signal_type == Signal.BUY:
                    # Check if we already have a long position
                    if position and position.is_long:
                        logger.info(f"Ignoring BUY signal for {symbol}: Already long")
                        continue
                    
                    # Close any existing short position
                    if position and position.is_short:
                        self.risk_manager.close_all_positions(account_id)
                    
                    # Calculate position size
                    size = self.risk_manager.calculate_position_size(
                        account_id=account_id,
                        contract_id=contract_id,
                        entry_price=signal.price
                    )
                    
                    if size <= 0:
                        logger.warning(f"Ignoring BUY signal for {symbol}: Position size is zero")
                        continue
                    
                    # Create order
                    order = Order(
                        account_id=account_id,
                        contract_id=contract_id,
                        action=Order.ACTION_BUY,
                        order_type=Order.TYPE_MARKET,
                        quantity=size,
                        time_in_force=Order.TIF_DAY
                    )
                    
                    # Place order
                    self.order_manager.place_order(order)
                    
                    logger.info(f"Executed BUY signal for {symbol}: {size} @ {signal.price}")
                    
                elif signal.signal_type == Signal.SELL:
                    # Check if we already have a short position
                    if position and position.is_short:
                        logger.info(f"Ignoring SELL signal for {symbol}: Already short")
                        continue
                    
                    # Close any existing long position
                    if position and position.is_long:
                        self.risk_manager.close_all_positions(account_id)
                    
                    # Calculate position size
                    size = self.risk_manager.calculate_position_size(
                        account_id=account_id,
                        contract_id=contract_id,
                        entry_price=signal.price
                    )
                    
                    if size <= 0:
                        logger.warning(f"Ignoring SELL signal for {symbol}: Position size is zero")
                        continue
                    
                    # Create order
                    order = Order(
                        account_id=account_id,
                        contract_id=contract_id,
                        action=Order.ACTION_SELL,
                        order_type=Order.TYPE_MARKET,
                        quantity=size,
                        time_in_force=Order.TIF_DAY
                    )
                    
                    # Place order
                    self.order_manager.place_order(order)
                    
                    logger.info(f"Executed SELL signal for {symbol}: {size} @ {signal.price}")
                    
                elif signal.signal_type in [Signal.CLOSE_LONG, Signal.CLOSE_SHORT]:
                    # Check if we have a position to close
                    if not position or position.net_pos == 0:
                        logger.info(f"Ignoring CLOSE signal for {symbol}: No position")
                        continue
                    
                    # Check if signal matches position direction
                    if (signal.signal_type == Signal.CLOSE_LONG and not position.is_long) or \
                       (signal.signal_type == Signal.CLOSE_SHORT and not position.is_short):
                        logger.info(f"Ignoring CLOSE signal for {symbol}: Direction mismatch")
                        continue
                    
                    # Close position
                    self.risk_manager.close_all_positions(account_id)
                    
                    logger.info(f"Executed CLOSE signal for {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing signal for {symbol}: {str(e)}")
    
    def _process_simulation_signals(self, 
                                   symbol: str, 
                                   signals: List[Signal],
                                   current_bar: Any) -> None:
        """
        Process trading signals in simulation mode.
        
        Args:
            symbol: Trading symbol
            signals: List of signals
            current_bar: Current price bar
        """
        # In simulation mode, we just log the signals
        for signal in signals:
            logger.info(f"[SIMULATION] Signal for {symbol}: {signal.signal_type} @ {signal.price}")
    
    def _initialize_strategies(self) -> None:
        """Initialize trading strategies."""
        strategy_configs = self.config.get("strategies", [])
        
        for config in strategy_configs:
            try:
                # Get strategy type and symbol
                strategy_type = config.get("type")
                symbol = config.get("symbol")
                
                if not strategy_type or not symbol:
                    logger.error(f"Invalid strategy configuration: {config}")
                    continue
                
                # Add symbol
                if not self.add_symbol(symbol):
                    continue
                
                # Create strategy
                strategy = self.strategy_selector.create_strategy(
                    strategy_type=strategy_type,
                    name=config.get("name"),
                    parameters=config.get("parameters", {})
                )
                
                # Initialize strategy
                strategy.initialize()
                
                # Store strategy
                self.strategies[symbol] = strategy
                
                logger.info(f"Initialized {strategy_type} strategy for {symbol}")
                
            except Exception as e:
                logger.error(f"Error initializing strategy: {str(e)}")
    
    def _load_simulation_data(self, symbol: str) -> bool:
        """
        Load historical data for simulation.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if data was loaded successfully
        """
        try:
            # Check if we have a data file for this symbol
            data_dir = self.config.get("simulation_data_dir", "data")
            file_path = os.path.join(data_dir, f"{symbol}.csv")
            
            if not os.path.exists(file_path):
                logger.error(f"Simulation data file not found: {file_path}")
                return False
            
            # Load data
            import pandas as pd
            data = pd.read_csv(file_path)
            
            # Ensure required columns exist
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in required_columns:
                if col not in data.columns:
                    logger.error(f"Missing required column in simulation data: {col}")
                    return False
            
            # Convert timestamp to datetime
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data.set_index("timestamp", inplace=True)
            
            # Store data
            self.simulation_data[symbol] = data
            self.simulation_index[symbol] = 0
            
            logger.info(f"Loaded simulation data for {symbol}: {len(data)} bars")
            return True
            
        except Exception as e:
            logger.error(f"Error loading simulation data for {symbol}: {str(e)}")
            return False
    
    def _convert_to_strategy_market_data(self, symbol: str, bar: Any) -> StrategyMarketData:
        """
        Convert a price bar to strategy market data.
        
        Args:
            symbol: Trading symbol
            bar: Price bar
            
        Returns:
            Strategy market data
        """
        return StrategyMarketData(
            symbol=symbol,
            timestamp=bar.name if hasattr(bar, "name") else datetime.now(),
            open=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            volume=bar["volume"]
        )
    
    def _get_symbol_for_contract(self, contract_id: int) -> Optional[str]:
        """
        Get symbol for a contract ID.
        
        Args:
            contract_id: Contract ID
            
        Returns:
            Symbol or None if not found
        """
        try:
            contract_info = self.account_manager.get_contract_info(contract_id)
            if contract_info:
                return contract_info.get("name")
            return None
        except Exception:
            return None
    
    def _log_status(self) -> None:
        """Log current status."""
        try:
            status = self.get_status()
            
            # Log basic status
            logger.info(f"Status: {status['mode']} mode, running for {status['runtime']}")
            
            # Log account info
            account = status["account"]
            if account["balance"] is not None:
                logger.info(f"Account: ${account['balance']:.2f}, P&L: ${account['pnl']:.2f} ({account['pnl_percentage']:.2f}%)")
            
            # Log positions
            if status["positions"]:
                for pos in status["positions"]:
                    logger.info(f"Position: {pos['symbol']} {pos['direction']} {pos['size']} @ {pos['entry_price']:.2f}, P&L: ${pos['pnl']:.2f}")
            else:
                logger.info("No open positions")
            
        except Exception as e:
            logger.error(f"Error logging status: {str(e)}")
    
    def _create_api_client(self) -> TradeovateApiClient:
        """
        Create Tradeovate API client from configuration.
        
        Returns:
            API client
        """
        api_config = self.config.get("api", {})
        
        return TradeovateApiClient(
            username=api_config.get("username", ""),
            password=api_config.get("password", ""),
            app_id=api_config.get("app_id", ""),
            app_version=api_config.get("app_version", "1.0"),
            cid=api_config.get("cid", 0),
            sec=api_config.get("sec", ""),
            device_id=api_config.get("device_id"),
            use_live=api_config.get("use_live", False),
            token_refresh_margin=api_config.get("token_refresh_margin", 300),
            max_retries=api_config.get("max_retries", 3),
            retry_delay=api_config.get("retry_delay", 2)
        )
    
    def _create_risk_parameters(self) -> RiskParameters:
        """
        Create risk parameters from configuration.
        
        Returns:
            Risk parameters
        """
        risk_config = self.config.get("risk", {})
        
        return RiskParameters(
            max_position_size=risk_config.get("max_position_size", 1),
            max_positions=risk_config.get("max_positions", 3),
            max_drawdown_pct=risk_config.get("max_drawdown_pct", 5.0),
            max_daily_loss_pct=risk_config.get("max_daily_loss_pct", 2.0),
            max_trade_risk_pct=risk_config.get("max_trade_risk_pct", 1.0),
            position_size_method=risk_config.get("position_size_method", "fixed"),
            kelly_fraction=risk_config.get("kelly_fraction", 0.5),
            use_stop_loss=risk_config.get("use_stop_loss", True),
            stop_loss_atr_multiple=risk_config.get("stop_loss_atr_multiple", 2.0),
            fixed_stop_loss_pct=risk_config.get("fixed_stop_loss_pct", 1.0),
            use_trailing_stop=risk_config.get("use_trailing_stop", True),
            trailing_stop_atr_multiple=risk_config.get("trailing_stop_atr_multiple", 1.5),
            fixed_trailing_stop_pct=risk_config.get("fixed_trailing_stop_pct", 0.75),
            use_take_profit=risk_config.get("use_take_profit", True),
            take_profit_atr_multiple=risk_config.get("take_profit_atr_multiple", 3.0),
            fixed_take_profit_pct=risk_config.get("fixed_take_profit_pct", 2.0),
            max_open_time_minutes=risk_config.get("max_open_time_minutes", 240)
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
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
            raise Exception(f"Configuration file not found: {config_path}")
        
        # Determine file format
        if config_path.endswith(".json"):
            with open(config_path, "r") as f:
                return json.load(f)
        elif config_path.endswith((".yaml", ".yml")):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            raise Exception(f"Unsupported configuration file format: {config_path}")
