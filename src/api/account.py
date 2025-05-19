"""
Account management module for Tradeovate API.
"""
from typing import Dict, Any, List, Optional, Union, Callable
import logging
from datetime import datetime

from .client import TradeovateApiClient, TradeovateApiError

logger = logging.getLogger(__name__)

class Account:
    """
    Represents a trading account in the system.
    """
    
    def __init__(self, 
                 account_id: int,
                 name: str,
                 user_id: int,
                 account_type: str,
                 active: bool = True,
                 cash_balance: float = 0.0,
                 initial_balance: float = 0.0,
                 buying_power: Optional[float] = None,
                 risk_discount_factor: float = 1.0,
                 auto_liq_level: float = 0.0,
                 trailing_max_drawdown: float = 0.0,
                 max_day_trades: int = 0,
                 day_trade_count: int = 0,
                 last_update_time: Optional[datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an account.
        
        Args:
            account_id: Account ID
            name: Account name
            user_id: User ID
            account_type: Account type
            active: Whether account is active
            cash_balance: Current cash balance
            initial_balance: Initial cash balance
            buying_power: Available buying power
            risk_discount_factor: Risk discount factor
            auto_liq_level: Auto-liquidation level
            trailing_max_drawdown: Trailing maximum drawdown
            max_day_trades: Maximum allowed day trades
            day_trade_count: Current day trade count
            last_update_time: Last update time
            metadata: Additional account metadata
        """
        self.account_id = account_id
        self.name = name
        self.user_id = user_id
        self.account_type = account_type
        self.active = active
        self.cash_balance = cash_balance
        self.initial_balance = initial_balance
        self.buying_power = buying_power
        self.risk_discount_factor = risk_discount_factor
        self.auto_liq_level = auto_liq_level
        self.trailing_max_drawdown = trailing_max_drawdown
        self.max_day_trades = max_day_trades
        self.day_trade_count = day_trade_count
        self.last_update_time = last_update_time or datetime.now()
        self.metadata = metadata or {}
    
    @property
    def pnl(self) -> float:
        """Calculate profit/loss."""
        return self.cash_balance - self.initial_balance
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate profit/loss percentage."""
        if self.initial_balance == 0:
            return 0.0
        return (self.pnl / self.initial_balance) * 100
    
    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'Account':
        """
        Create an account from API response.
        
        Args:
            response: API response
            
        Returns:
            Account object
        """
        # Map API response fields to Account constructor parameters
        account_id = response.get("id")
        name = response.get("name", "")
        user_id = response.get("userId", 0)
        account_type = response.get("accountType", "")
        active = response.get("active", True)
        
        # Create account object
        account = cls(
            account_id=account_id,
            name=name,
            user_id=user_id,
            account_type=account_type,
            active=active
        )
        
        return account
    
    def __str__(self) -> str:
        """String representation of the account."""
        return f"Account(id={self.account_id}, name={self.name}, balance=${self.cash_balance:.2f})"

class Position:
    """
    Represents a trading position in the system.
    """
    
    def __init__(self, 
                 account_id: int,
                 contract_id: int,
                 net_pos: int,
                 net_price: float,
                 timestamp: Optional[datetime] = None,
                 p_and_l: float = 0.0,
                 open_p_and_l: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a position.
        
        Args:
            account_id: Account ID
            contract_id: Contract ID
            net_pos: Net position (positive for long, negative for short)
            net_price: Average entry price
            timestamp: Position timestamp
            p_and_l: Realized profit/loss
            open_p_and_l: Unrealized profit/loss
            metadata: Additional position metadata
        """
        self.account_id = account_id
        self.contract_id = contract_id
        self.net_pos = net_pos
        self.net_price = net_price
        self.timestamp = timestamp or datetime.now()
        self.p_and_l = p_and_l
        self.open_p_and_l = open_p_and_l
        self.metadata = metadata or {}
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.net_pos > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.net_pos < 0
    
    @property
    def size(self) -> int:
        """Get absolute position size."""
        return abs(self.net_pos)
    
    @property
    def total_pnl(self) -> float:
        """Get total profit/loss (realized + unrealized)."""
        return self.p_and_l + self.open_p_and_l
    
    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'Position':
        """
        Create a position from API response.
        
        Args:
            response: API response
            
        Returns:
            Position object
        """
        # Map API response fields to Position constructor parameters
        account_id = response.get("accountId")
        contract_id = response.get("contractId")
        net_pos = response.get("netPos", 0)
        net_price = response.get("netPrice", 0.0)
        p_and_l = response.get("pnl", 0.0)
        open_p_and_l = response.get("openPnl", 0.0)
        
        # Parse timestamp
        timestamp = None
        if "timestamp" in response:
            timestamp = datetime.fromtimestamp(response["timestamp"] / 1000)  # Convert ms to seconds
        
        return cls(
            account_id=account_id,
            contract_id=contract_id,
            net_pos=net_pos,
            net_price=net_price,
            timestamp=timestamp,
            p_and_l=p_and_l,
            open_p_and_l=open_p_and_l
        )
    
    def __str__(self) -> str:
        """String representation of the position."""
        direction = "LONG" if self.is_long else "SHORT"
        return f"Position({direction} {self.size} @ {self.net_price:.2f}, P&L=${self.total_pnl:.2f})"

class AccountManager:
    """
    Manages accounts and positions for the trading system.
    
    This class handles account information, position tracking,
    and risk management.
    """
    
    def __init__(self, api_client: TradeovateApiClient):
        """
        Initialize the account manager.
        
        Args:
            api_client: Tradeovate API client
        """
        self.api_client = api_client
        self.accounts = {}  # account_id -> Account
        self.positions = {}  # (account_id, contract_id) -> Position
        self.contracts = {}  # contract_id -> Contract info
    
    def get_accounts(self) -> List[Account]:
        """
        Get all accounts.
        
        Returns:
            List of accounts
        """
        try:
            # Fetch accounts from API
            api_accounts = self.api_client.get_accounts()
            
            # Process each account
            for api_account in api_accounts:
                account_id = api_account.get("id")
                if account_id:
                    if account_id in self.accounts:
                        # Update existing account
                        self._update_account_from_api(self.accounts[account_id], api_account)
                    else:
                        # Create new account
                        account = Account.from_api_response(api_account)
                        self.accounts[account_id] = account
            
            return list(self.accounts.values())
            
        except Exception as e:
            logger.error(f"Error fetching accounts: {str(e)}")
            return list(self.accounts.values())
    
    def get_account(self, account_id: int) -> Optional[Account]:
        """
        Get an account by ID.
        
        Args:
            account_id: Account ID
            
        Returns:
            Account or None if not found
        """
        # Check if we already have this account
        if account_id in self.accounts:
            return self.accounts[account_id]
        
        try:
            # Fetch account from API
            api_account = self.api_client.get_account_info(account_id)
            
            # Create account object
            account = Account.from_api_response(api_account)
            self.accounts[account_id] = account
            
            return account
            
        except Exception as e:
            logger.error(f"Error fetching account {account_id}: {str(e)}")
            return None
    
    def get_positions(self, account_id: int) -> List[Position]:
        """
        Get positions for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            List of positions
        """
        try:
            # Fetch positions from API
            api_positions = self.api_client.get_positions(account_id)
            
            # Process each position
            positions = []
            for api_position in api_positions:
                contract_id = api_position.get("contractId")
                if contract_id:
                    position_key = (account_id, contract_id)
                    
                    if position_key in self.positions:
                        # Update existing position
                        self._update_position_from_api(self.positions[position_key], api_position)
                        positions.append(self.positions[position_key])
                    else:
                        # Create new position
                        position = Position.from_api_response(api_position)
                        self.positions[position_key] = position
                        positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error fetching positions for account {account_id}: {str(e)}")
            
            # Return positions we already have for this account
            return [
                position for (acc_id, _), position in self.positions.items()
                if acc_id == account_id
            ]
    
    def get_position(self, account_id: int, contract_id: int) -> Optional[Position]:
        """
        Get a specific position.
        
        Args:
            account_id: Account ID
            contract_id: Contract ID
            
        Returns:
            Position or None if not found
        """
        position_key = (account_id, contract_id)
        
        # Check if we already have this position
        if position_key in self.positions:
            return self.positions[position_key]
        
        # Try to fetch positions for this account
        positions = self.get_positions(account_id)
        
        # Find the specific position
        for position in positions:
            if position.contract_id == contract_id:
                return position
        
        return None
    
    def get_account_risk_status(self, account_id: int) -> Dict[str, Any]:
        """
        Get risk status for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Risk status information
        """
        try:
            return self.api_client.get_account_risk_status(account_id)
        except Exception as e:
            logger.error(f"Error fetching risk status for account {account_id}: {str(e)}")
            return {}
    
    def get_cash_balance(self, account_id: int) -> float:
        """
        Get cash balance for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Cash balance
        """
        try:
            # Fetch cash balance from API
            balance_info = self.api_client.get_cash_balance(account_id)
            
            # Extract balance
            cash_balance = balance_info.get("cashBalance", 0.0)
            
            # Update account if we have it
            if account_id in self.accounts:
                self.accounts[account_id].cash_balance = cash_balance
                self.accounts[account_id].last_update_time = datetime.now()
            
            return cash_balance
            
        except Exception as e:
            logger.error(f"Error fetching cash balance for account {account_id}: {str(e)}")
            
            # Return cached balance if available
            if account_id in self.accounts:
                return self.accounts[account_id].cash_balance
            
            return 0.0
    
    def get_contract_info(self, contract_id: int) -> Dict[str, Any]:
        """
        Get contract information.
        
        Args:
            contract_id: Contract ID
            
        Returns:
            Contract information
        """
        # Check if we already have this contract
        if contract_id in self.contracts:
            return self.contracts[contract_id]
        
        try:
            # Fetch contract from API
            contract_info = self.api_client.get_contract(contract_id)
            
            # Cache contract info
            self.contracts[contract_id] = contract_info
            
            return contract_info
            
        except Exception as e:
            logger.error(f"Error fetching contract {contract_id}: {str(e)}")
            return {}
    
    def find_contract(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a contract by name.
        
        Args:
            name: Contract name
            
        Returns:
            Contract information or None if not found
        """
        try:
            # Search for contract
            contracts = self.api_client.find_contracts(name)
            
            if contracts:
                # Cache contract info
                contract = contracts[0]
                contract_id = contract.get("id")
                if contract_id:
                    self.contracts[contract_id] = contract
                
                return contract
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding contract {name}: {str(e)}")
            return None
    
    def update_account_info(self, account_id: int) -> bool:
        """
        Update account information from API.
        
        Args:
            account_id: Account ID
            
        Returns:
            True if update was successful
        """
        try:
            # Fetch account info
            api_account = self.api_client.get_account_info(account_id)
            
            # Update or create account
            if account_id in self.accounts:
                self._update_account_from_api(self.accounts[account_id], api_account)
            else:
                account = Account.from_api_response(api_account)
                self.accounts[account_id] = account
            
            # Update cash balance
            self.get_cash_balance(account_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating account {account_id}: {str(e)}")
            return False
    
    def update_positions(self, account_id: int) -> bool:
        """
        Update positions for an account from API.
        
        Args:
            account_id: Account ID
            
        Returns:
            True if update was successful
        """
        try:
            # Fetch positions
            self.get_positions(account_id)
            return True
            
        except Exception as e:
            logger.error(f"Error updating positions for account {account_id}: {str(e)}")
            return False
    
    def _update_account_from_api(self, account: Account, api_data: Dict[str, Any]) -> None:
        """
        Update an account with API data.
        
        Args:
            account: Account to update
            api_data: API data
        """
        # Update account fields
        if "name" in api_data:
            account.name = api_data["name"]
        
        if "active" in api_data:
            account.active = api_data["active"]
        
        if "accountType" in api_data:
            account.account_type = api_data["accountType"]
        
        account.last_update_time = datetime.now()
    
    def _update_position_from_api(self, position: Position, api_data: Dict[str, Any]) -> None:
        """
        Update a position with API data.
        
        Args:
            position: Position to update
            api_data: API data
        """
        # Update position fields
        if "netPos" in api_data:
            position.net_pos = api_data["netPos"]
        
        if "netPrice" in api_data:
            position.net_price = api_data["netPrice"]
        
        if "pnl" in api_data:
            position.p_and_l = api_data["pnl"]
        
        if "openPnl" in api_data:
            position.open_p_and_l = api_data["openPnl"]
        
        if "timestamp" in api_data:
            position.timestamp = datetime.fromtimestamp(api_data["timestamp"] / 1000)
        else:
            position.timestamp = datetime.now()
