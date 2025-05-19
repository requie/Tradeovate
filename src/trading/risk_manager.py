"""
Risk management module for the trading bot.
"""
from typing import Dict, Any, List, Optional, Union, Callable
import logging
import numpy as np
from datetime import datetime, timedelta

from ..api.account import Account, Position, AccountManager
from ..api.order import Order, OrderManager
from ..api.market_data import MarketData

logger = logging.getLogger(__name__)

class RiskParameters:
    """Configuration parameters for risk management."""
    
    def __init__(self,
                 max_position_size: int = 1,
                 max_positions: int = 3,
                 max_drawdown_pct: float = 5.0,
                 max_daily_loss_pct: float = 2.0,
                 max_trade_risk_pct: float = 1.0,
                 position_size_method: str = "fixed",
                 kelly_fraction: float = 0.5,
                 use_stop_loss: bool = True,
                 stop_loss_atr_multiple: float = 2.0,
                 fixed_stop_loss_pct: float = 1.0,
                 use_trailing_stop: bool = True,
                 trailing_stop_atr_multiple: float = 1.5,
                 fixed_trailing_stop_pct: float = 0.75,
                 use_take_profit: bool = True,
                 take_profit_atr_multiple: float = 3.0,
                 fixed_take_profit_pct: float = 2.0,
                 max_open_time_minutes: int = 240):
        """
        Initialize risk parameters.
        
        Args:
            max_position_size: Maximum position size in contracts
            max_positions: Maximum number of open positions
            max_drawdown_pct: Maximum drawdown percentage allowed
            max_daily_loss_pct: Maximum daily loss percentage allowed
            max_trade_risk_pct: Maximum risk percentage per trade
            position_size_method: Method for position sizing ("fixed", "percent_risk", "kelly")
            kelly_fraction: Fraction of Kelly criterion to use (0.0 to 1.0)
            use_stop_loss: Whether to use stop loss
            stop_loss_atr_multiple: Stop loss as multiple of ATR
            fixed_stop_loss_pct: Fixed stop loss percentage
            use_trailing_stop: Whether to use trailing stop
            trailing_stop_atr_multiple: Trailing stop as multiple of ATR
            fixed_trailing_stop_pct: Fixed trailing stop percentage
            use_take_profit: Whether to use take profit
            take_profit_atr_multiple: Take profit as multiple of ATR
            fixed_take_profit_pct: Fixed take profit percentage
            max_open_time_minutes: Maximum time to keep a position open in minutes
        """
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_trade_risk_pct = max_trade_risk_pct
        self.position_size_method = position_size_method
        self.kelly_fraction = kelly_fraction
        self.use_stop_loss = use_stop_loss
        self.stop_loss_atr_multiple = stop_loss_atr_multiple
        self.fixed_stop_loss_pct = fixed_stop_loss_pct
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_atr_multiple = trailing_stop_atr_multiple
        self.fixed_trailing_stop_pct = fixed_trailing_stop_pct
        self.use_take_profit = use_take_profit
        self.take_profit_atr_multiple = take_profit_atr_multiple
        self.fixed_take_profit_pct = fixed_take_profit_pct
        self.max_open_time_minutes = max_open_time_minutes

class RiskManager:
    """
    Manages risk for the trading system.
    
    This class handles position sizing, stop loss, take profit,
    and overall risk exposure management.
    """
    
    def __init__(self, 
                 account_manager: AccountManager,
                 order_manager: OrderManager,
                 market_data: MarketData,
                 risk_parameters: Optional[RiskParameters] = None):
        """
        Initialize the risk manager.
        
        Args:
            account_manager: Account manager
            order_manager: Order manager
            market_data: Market data handler
            risk_parameters: Risk parameters (uses defaults if None)
        """
        self.account_manager = account_manager
        self.order_manager = order_manager
        self.market_data = market_data
        self.risk_parameters = risk_parameters or RiskParameters()
        
        # Performance tracking
        self.initial_equity = {}  # account_id -> initial equity
        self.peak_equity = {}     # account_id -> peak equity
        self.daily_starting_equity = {}  # account_id -> daily starting equity
        self.trade_history = []   # List of completed trades
        
        # Risk state
        self.is_max_drawdown_breached = False
        self.is_daily_loss_breached = False
        self.last_equity_check = datetime.now()
    
    def calculate_position_size(self, 
                               account_id: int,
                               contract_id: int,
                               entry_price: float,
                               stop_price: Optional[float] = None,
                               win_rate: Optional[float] = None,
                               reward_risk_ratio: Optional[float] = None) -> int:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            account_id: Account ID
            contract_id: Contract ID
            entry_price: Entry price
            stop_price: Stop loss price (required for percent_risk method)
            win_rate: Historical win rate (required for Kelly method)
            reward_risk_ratio: Reward to risk ratio (required for Kelly method)
            
        Returns:
            Position size in contracts
        """
        # Get account information
        account = self.account_manager.get_account(account_id)
        if not account:
            logger.error(f"Cannot calculate position size: Account {account_id} not found")
            return 0
        
        # Get contract information
        contract_info = self.account_manager.get_contract_info(contract_id)
        if not contract_info:
            logger.error(f"Cannot calculate position size: Contract {contract_id} not found")
            return 0
        
        # Get current positions
        positions = self.account_manager.get_positions(account_id)
        
        # Check if we've reached maximum positions
        if len(positions) >= self.risk_parameters.max_positions:
            logger.warning(f"Maximum positions ({self.risk_parameters.max_positions}) reached")
            return 0
        
        # Calculate position size based on method
        if self.risk_parameters.position_size_method == "fixed":
            # Fixed position size
            position_size = self.risk_parameters.max_position_size
            
        elif self.risk_parameters.position_size_method == "percent_risk":
            # Position size based on percentage risk
            if stop_price is None:
                logger.error("Stop price is required for percent_risk position sizing")
                return 0
            
            # Calculate risk per contract
            risk_per_contract = abs(entry_price - stop_price)
            if risk_per_contract <= 0:
                logger.error("Invalid risk per contract (entry price and stop price are equal)")
                return 0
            
            # Calculate maximum risk amount
            max_risk_amount = account.cash_balance * (self.risk_parameters.max_trade_risk_pct / 100)
            
            # Calculate position size
            position_size = int(max_risk_amount / risk_per_contract)
            
        elif self.risk_parameters.position_size_method == "kelly":
            # Kelly criterion position sizing
            if win_rate is None or reward_risk_ratio is None:
                logger.error("Win rate and reward/risk ratio are required for Kelly position sizing")
                return 0
            
            # Calculate Kelly fraction
            kelly = win_rate - ((1 - win_rate) / reward_risk_ratio)
            
            # Apply Kelly fraction and limit to positive values
            kelly = max(0, kelly * self.risk_parameters.kelly_fraction)
            
            # Calculate position size based on Kelly percentage of account
            max_risk_amount = account.cash_balance * kelly
            
            # Calculate risk per contract
            if stop_price is not None:
                risk_per_contract = abs(entry_price - stop_price)
                if risk_per_contract <= 0:
                    logger.error("Invalid risk per contract (entry price and stop price are equal)")
                    return 0
                
                # Calculate position size
                position_size = int(max_risk_amount / risk_per_contract)
            else:
                # Use a default risk percentage if stop price not provided
                default_risk_pct = self.risk_parameters.fixed_stop_loss_pct / 100
                risk_per_contract = entry_price * default_risk_pct
                position_size = int(max_risk_amount / risk_per_contract)
        
        else:
            logger.error(f"Unknown position sizing method: {self.risk_parameters.position_size_method}")
            return 0
        
        # Limit to maximum position size
        position_size = min(position_size, self.risk_parameters.max_position_size)
        
        # Ensure position size is at least 1
        position_size = max(1, position_size)
        
        logger.info(f"Calculated position size: {position_size} contracts")
        return position_size
    
    def calculate_stop_loss(self,
                           entry_price: float,
                           direction: str,  # "long" or "short"
                           atr: Optional[float] = None) -> float:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Entry price
            direction: Trade direction ("long" or "short")
            atr: Average True Range (optional)
            
        Returns:
            Stop loss price
        """
        if not self.risk_parameters.use_stop_loss:
            return 0.0
        
        if atr is not None and atr > 0:
            # ATR-based stop loss
            stop_distance = atr * self.risk_parameters.stop_loss_atr_multiple
        else:
            # Percentage-based stop loss
            stop_distance = entry_price * (self.risk_parameters.fixed_stop_loss_pct / 100)
        
        if direction.lower() == "long":
            stop_price = entry_price - stop_distance
        else:  # short
            stop_price = entry_price + stop_distance
        
        return stop_price
    
    def calculate_take_profit(self,
                             entry_price: float,
                             direction: str,  # "long" or "short"
                             atr: Optional[float] = None) -> float:
        """
        Calculate take profit price.
        
        Args:
            entry_price: Entry price
            direction: Trade direction ("long" or "short")
            atr: Average True Range (optional)
            
        Returns:
            Take profit price
        """
        if not self.risk_parameters.use_take_profit:
            return 0.0
        
        if atr is not None and atr > 0:
            # ATR-based take profit
            profit_distance = atr * self.risk_parameters.take_profit_atr_multiple
        else:
            # Percentage-based take profit
            profit_distance = entry_price * (self.risk_parameters.fixed_take_profit_pct / 100)
        
        if direction.lower() == "long":
            take_profit_price = entry_price + profit_distance
        else:  # short
            take_profit_price = entry_price - profit_distance
        
        return take_profit_price
    
    def update_trailing_stop(self,
                            position: Position,
                            current_price: float,
                            current_stop: float,
                            atr: Optional[float] = None) -> float:
        """
        Update trailing stop price.
        
        Args:
            position: Position
            current_price: Current market price
            current_stop: Current stop price
            atr: Average True Range (optional)
            
        Returns:
            Updated stop price
        """
        if not self.risk_parameters.use_trailing_stop:
            return current_stop
        
        if atr is not None and atr > 0:
            # ATR-based trailing stop
            stop_distance = atr * self.risk_parameters.trailing_stop_atr_multiple
        else:
            # Percentage-based trailing stop
            stop_distance = current_price * (self.risk_parameters.fixed_trailing_stop_pct / 100)
        
        if position.is_long:
            # For long positions, move stop up if price increases
            new_stop = current_price - stop_distance
            if new_stop > current_stop:
                return new_stop
        else:  # short
            # For short positions, move stop down if price decreases
            new_stop = current_price + stop_distance
            if new_stop < current_stop:
                return new_stop
        
        return current_stop
    
    def check_risk_limits(self, account_id: int) -> bool:
        """
        Check if current positions exceed risk limits.
        
        Args:
            account_id: Account ID
            
        Returns:
            True if risk limits are not exceeded, False otherwise
        """
        # Get account information
        account = self.account_manager.get_account(account_id)
        if not account:
            logger.error(f"Cannot check risk limits: Account {account_id} not found")
            return False
        
        # Initialize tracking if needed
        if account_id not in self.initial_equity:
            self.initial_equity[account_id] = account.cash_balance
            self.peak_equity[account_id] = account.cash_balance
        
        if account_id not in self.daily_starting_equity:
            self.daily_starting_equity[account_id] = account.cash_balance
        
        # Check if we need to reset daily tracking
        now = datetime.now()
        if now.date() > self.last_equity_check.date():
            self.daily_starting_equity[account_id] = account.cash_balance
            self.is_daily_loss_breached = False
        
        self.last_equity_check = now
        
        # Update peak equity if current equity is higher
        if account.cash_balance > self.peak_equity[account_id]:
            self.peak_equity[account_id] = account.cash_balance
        
        # Calculate drawdown
        if self.peak_equity[account_id] > 0:
            current_drawdown_pct = ((self.peak_equity[account_id] - account.cash_balance) / 
                                    self.peak_equity[account_id]) * 100
            
            # Check if drawdown exceeds limit
            if current_drawdown_pct > self.risk_parameters.max_drawdown_pct:
                if not self.is_max_drawdown_breached:
                    logger.warning(f"Maximum drawdown exceeded: {current_drawdown_pct:.2f}% > {self.risk_parameters.max_drawdown_pct:.2f}%")
                    self.is_max_drawdown_breached = True
                return False
        
        # Calculate daily loss
        if self.daily_starting_equity[account_id] > 0:
            daily_loss_pct = ((self.daily_starting_equity[account_id] - account.cash_balance) / 
                             self.daily_starting_equity[account_id]) * 100
            
            # Check if daily loss exceeds limit
            if daily_loss_pct > self.risk_parameters.max_daily_loss_pct:
                if not self.is_daily_loss_breached:
                    logger.warning(f"Maximum daily loss exceeded: {daily_loss_pct:.2f}% > {self.risk_parameters.max_daily_loss_pct:.2f}%")
                    self.is_daily_loss_breached = True
                return False
        
        # All checks passed
        return True
    
    def manage_position_risk(self, 
                            account_id: int, 
                            contract_id: int,
                            current_price: float) -> None:
        """
        Manage risk for an open position.
        
        This includes updating trailing stops, checking time limits,
        and other risk management actions.
        
        Args:
            account_id: Account ID
            contract_id: Contract ID
            current_price: Current market price
        """
        # Get position
        position = self.account_manager.get_position(account_id, contract_id)
        if not position or position.net_pos == 0:
            return
        
        # Get active orders for this position
        active_orders = [
            order for order in self.order_manager.get_orders(account_id, active_only=True)
            if order.contract_id == contract_id
        ]
        
        # Find stop loss orders
        stop_orders = [
            order for order in active_orders
            if order.order_type in [Order.TYPE_STOP, Order.TYPE_STOP_LIMIT]
        ]
        
        # Check if we need to add a stop loss
        if self.risk_parameters.use_stop_loss and not stop_orders:
            # Calculate stop loss price
            direction = "long" if position.is_long else "short"
            
            # Try to get ATR from market data
            atr = self._get_atr_for_contract(contract_id)
            
            stop_price = self.calculate_stop_loss(
                entry_price=position.net_price,
                direction=direction,
                atr=atr
            )
            
            # Place stop order
            if stop_price > 0:
                self._place_stop_order(
                    account_id=account_id,
                    contract_id=contract_id,
                    stop_price=stop_price,
                    quantity=abs(position.net_pos),
                    is_buy=(not position.is_long)  # Buy to close short, Sell to close long
                )
        
        # Update trailing stops if needed
        elif self.risk_parameters.use_trailing_stop and stop_orders:
            # Try to get ATR from market data
            atr = self._get_atr_for_contract(contract_id)
            
            for stop_order in stop_orders:
                # Calculate new stop price
                new_stop = self.update_trailing_stop(
                    position=position,
                    current_price=current_price,
                    current_stop=stop_order.stop_price,
                    atr=atr
                )
                
                # Update stop order if needed
                if new_stop != stop_order.stop_price:
                    try:
                        self.order_manager.modify_order(
                            order_id=stop_order.order_id,
                            stop_price=new_stop
                        )
                        logger.info(f"Updated trailing stop for position {position} to {new_stop}")
                    except Exception as e:
                        logger.error(f"Error updating trailing stop: {str(e)}")
        
        # Check if position has been open too long
        if position.timestamp:
            position_age = datetime.now() - position.timestamp
            max_age = timedelta(minutes=self.risk_parameters.max_open_time_minutes)
            
            if position_age > max_age:
                logger.info(f"Position {position} has exceeded maximum age ({self.risk_parameters.max_open_time_minutes} minutes), closing")
                
                # Close position with market order
                self._close_position_with_market(
                    account_id=account_id,
                    contract_id=contract_id,
                    position=position
                )
    
    def close_all_positions(self, account_id: int) -> int:
        """
        Close all positions for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Number of positions closed
        """
        # Get positions
        positions = self.account_manager.get_positions(account_id)
        
        # Close each position
        closed_count = 0
        for position in positions:
            if position.net_pos != 0:
                try:
                    self._close_position_with_market(
                        account_id=account_id,
                        contract_id=position.contract_id,
                        position=position
                    )
                    closed_count += 1
                except Exception as e:
                    logger.error(f"Error closing position {position}: {str(e)}")
        
        # Cancel all orders
        self.order_manager.cancel_all_orders(account_id)
        
        logger.info(f"Closed {closed_count} positions for account {account_id}")
        return closed_count
    
    def _place_stop_order(self, 
                         account_id: int,
                         contract_id: int,
                         stop_price: float,
                         quantity: int,
                         is_buy: bool) -> Optional[Order]:
        """
        Place a stop order.
        
        Args:
            account_id: Account ID
            contract_id: Contract ID
            stop_price: Stop price
            quantity: Order quantity
            is_buy: Whether this is a buy order
            
        Returns:
            Order object or None if placement failed
        """
        try:
            # Create order
            order = Order(
                account_id=account_id,
                contract_id=contract_id,
                action=Order.ACTION_BUY if is_buy else Order.ACTION_SELL,
                order_type=Order.TYPE_STOP,
                quantity=quantity,
                stop_price=stop_price,
                time_in_force=Order.TIF_GTC
            )
            
            # Place order
            client_order_id = self.order_manager.place_order(order)
            
            logger.info(f"Placed stop order: {order}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing stop order: {str(e)}")
            return None
    
    def _close_position_with_market(self, 
                                   account_id: int,
                                   contract_id: int,
                                   position: Position) -> Optional[Order]:
        """
        Close a position with a market order.
        
        Args:
            account_id: Account ID
            contract_id: Contract ID
            position: Position to close
            
        Returns:
            Order object or None if placement failed
        """
        try:
            # Create order
            order = Order(
                account_id=account_id,
                contract_id=contract_id,
                action=Order.ACTION_BUY if position.is_short else Order.ACTION_SELL,
                order_type=Order.TYPE_MARKET,
                quantity=abs(position.net_pos),
                time_in_force=Order.TIF_DAY
            )
            
            # Place order
            client_order_id = self.order_manager.place_order(order)
            
            logger.info(f"Closed position with market order: {position}")
            return order
            
        except Exception as e:
            logger.error(f"Error closing position with market order: {str(e)}")
            return None
    
    def _get_atr_for_contract(self, contract_id: int, period: int = 14) -> Optional[float]:
        """
        Calculate ATR for a contract.
        
        Args:
            contract_id: Contract ID
            period: ATR period
            
        Returns:
            ATR value or None if not available
        """
        try:
            # Get contract info to find symbol
            contract_info = self.account_manager.get_contract_info(contract_id)
            if not contract_info:
                return None
            
            symbol = contract_info.get("name")
            if not symbol:
                return None
            
            # Get historical data
            data = self.market_data.get_historical_data(symbol)
            if data.empty or len(data) < period:
                return None
            
            # Calculate ATR
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            
            tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
            atr = np.mean(tr[-period:])
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return None
