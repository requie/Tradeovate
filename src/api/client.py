"""
Tradeovate API client for trading bot.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import requests
import json
import time
import logging
import os
import uuid
from datetime import datetime, timedelta
import websocket
import threading
import queue

logger = logging.getLogger(__name__)

class TradeovateApiError(Exception):
    """Exception raised for Tradeovate API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)

class TradeovateApiClient:
    """
    Client for interacting with the Tradeovate API.
    
    This client handles authentication, token management, and provides
    methods for accessing various API endpoints.
    """
    
    # API endpoints
    DEMO_URL = "https://demo.tradovateapi.com/v1"
    LIVE_URL = "https://live.tradovateapi.com/v1"
    MD_URL = "https://md.tradovateapi.com/v1"
    
    # WebSocket endpoints
    DEMO_WS_URL = "wss://demo.tradovateapi.com/v1/websocket"
    LIVE_WS_URL = "wss://live.tradovateapi.com/v1/websocket"
    MD_WS_URL = "wss://md.tradovateapi.com/v1/websocket"
    
    def __init__(self, 
                 username: str,
                 password: str,
                 app_id: str,
                 app_version: str,
                 cid: int,
                 sec: str,
                 device_id: Optional[str] = None,
                 use_live: bool = False,
                 token_refresh_margin: int = 300,
                 max_retries: int = 3,
                 retry_delay: int = 2):
        """
        Initialize the Tradeovate API client.
        
        Args:
            username: Tradeovate account username
            password: Tradeovate account password
            app_id: Application identifier
            app_version: Application version
            cid: Client app ID provided by Tradeovate
            sec: Secret/API key provided by Tradeovate
            device_id: Unique device identifier (generated if not provided)
            use_live: Whether to use live trading (True) or demo (False)
            token_refresh_margin: Seconds before token expiry to refresh
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
        """
        self.username = username
        self.password = password
        self.app_id = app_id
        self.app_version = app_version
        self.cid = cid
        self.sec = sec
        self.device_id = device_id or str(uuid.uuid4())
        self.use_live = use_live
        self.token_refresh_margin = token_refresh_margin
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Token management
        self.access_token = None
        self.md_access_token = None
        self.token_expiry = None
        
        # WebSocket management
        self.ws = None
        self.md_ws = None
        self.ws_connected = False
        self.md_ws_connected = False
        self.ws_lock = threading.Lock()
        self.ws_thread = None
        self.md_ws_thread = None
        self.ws_queue = queue.Queue()
        self.md_ws_queue = queue.Queue()
        self.ws_callbacks = {}
        self.md_ws_callbacks = {}
        
        # Session for HTTP requests
        self.session = requests.Session()
        
        # Base URLs based on environment
        self.base_url = self.LIVE_URL if use_live else self.DEMO_URL
        self.md_url = self.MD_URL
        self.ws_url = self.LIVE_WS_URL if use_live else self.DEMO_WS_URL
        self.md_ws_url = self.MD_WS_URL
        
        logger.info(f"Initialized Tradeovate API client for {'live' if use_live else 'demo'} environment")
    
    def authenticate(self) -> Dict[str, Any]:
        """
        Authenticate with the Tradeovate API and obtain access tokens.
        
        Returns:
            Authentication response with tokens
        
        Raises:
            TradeovateApiError: If authentication fails
        """
        endpoint = "/auth/accesstokenrequest"
        
        payload = {
            "name": self.username,
            "password": self.password,
            "appId": self.app_id,
            "appVersion": self.app_version,
            "cid": self.cid,
            "sec": self.sec,
            "deviceId": self.device_id
        }
        
        try:
            response = self._make_request("POST", endpoint, payload)
            
            # Store tokens and expiry
            self.access_token = response.get("accessToken")
            self.md_access_token = response.get("mdAccessToken")
            
            # Parse expiry time
            expiry_str = response.get("expirationTime")
            if expiry_str:
                self.token_expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            
            logger.info("Successfully authenticated with Tradeovate API")
            return response
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise TradeovateApiError(f"Authentication failed: {str(e)}")
    
    def ensure_authenticated(self) -> None:
        """
        Ensure the client has valid authentication tokens.
        
        If tokens are missing or about to expire, this method will
        automatically re-authenticate.
        """
        # Check if we need to authenticate
        if (self.access_token is None or 
            self.token_expiry is None or 
            datetime.now(self.token_expiry.tzinfo) >= self.token_expiry - timedelta(seconds=self.token_refresh_margin)):
            
            logger.info("Authentication tokens missing or expiring soon, re-authenticating")
            self.authenticate()
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        Get information about the authenticated user.
        
        Returns:
            User information
        """
        self.ensure_authenticated()
        endpoint = "/auth/me"
        return self._make_request("GET", endpoint)
    
    def get_accounts(self) -> List[Dict[str, Any]]:
        """
        Get list of user accounts.
        
        Returns:
            List of accounts
        """
        self.ensure_authenticated()
        endpoint = "/account/list"
        return self._make_request("GET", endpoint)
    
    def get_account_info(self, account_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Account information
        """
        self.ensure_authenticated()
        endpoint = f"/account/item?id={account_id}"
        return self._make_request("GET", endpoint)
    
    def get_positions(self, account_id: int) -> List[Dict[str, Any]]:
        """
        Get current positions for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            List of positions
        """
        self.ensure_authenticated()
        endpoint = f"/position/list?accountId={account_id}"
        return self._make_request("GET", endpoint)
    
    def get_orders(self, account_id: int, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get orders for an account.
        
        Args:
            account_id: Account ID
            status: Optional order status filter
            
        Returns:
            List of orders
        """
        self.ensure_authenticated()
        endpoint = f"/order/list?accountId={account_id}"
        if status:
            endpoint += f"&status={status}"
        return self._make_request("GET", endpoint)
    
    def get_contract(self, contract_id: int) -> Dict[str, Any]:
        """
        Get contract details.
        
        Args:
            contract_id: Contract ID
            
        Returns:
            Contract details
        """
        self.ensure_authenticated()
        endpoint = f"/contract/item?id={contract_id}"
        return self._make_request("GET", endpoint)
    
    def find_contracts(self, name: str) -> List[Dict[str, Any]]:
        """
        Find contracts by name.
        
        Args:
            name: Contract name to search for
            
        Returns:
            List of matching contracts
        """
        self.ensure_authenticated()
        endpoint = f"/contract/find?name={name}"
        return self._make_request("GET", endpoint)
    
    def place_order(self, 
                   account_id: int,
                   contract_id: int,
                   action: str,  # "Buy" or "Sell"
                   order_type: str,  # "Market", "Limit", "Stop", "StopLimit"
                   quantity: int,
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   time_in_force: str = "Day",  # "Day", "GTC", "IOC", "FOK"
                   bracket_strategy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            account_id: Account ID
            contract_id: Contract ID
            action: Order action ("Buy" or "Sell")
            order_type: Order type ("Market", "Limit", "Stop", "StopLimit")
            quantity: Order quantity
            price: Limit price (required for Limit and StopLimit orders)
            stop_price: Stop price (required for Stop and StopLimit orders)
            time_in_force: Time in force
            bracket_strategy: Optional bracket order strategy
            
        Returns:
            Order response
        """
        self.ensure_authenticated()
        endpoint = "/order/placeorder"
        
        payload = {
            "accountId": account_id,
            "contractId": contract_id,
            "action": action,
            "orderQty": quantity,
            "orderType": order_type,
            "timeInForce": time_in_force
        }
        
        # Add price for Limit and StopLimit orders
        if order_type in ["Limit", "StopLimit"] and price is not None:
            payload["price"] = price
        
        # Add stop price for Stop and StopLimit orders
        if order_type in ["Stop", "StopLimit"] and stop_price is not None:
            payload["stopPrice"] = stop_price
        
        # Add bracket strategy if provided
        if bracket_strategy:
            payload["bracketStrategy"] = bracket_strategy
        
        return self._make_request("POST", endpoint, payload)
    
    def modify_order(self, 
                    order_id: int,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    quantity: Optional[int] = None,
                    time_in_force: Optional[str] = None) -> Dict[str, Any]:
        """
        Modify an existing order.
        
        Args:
            order_id: Order ID
            price: New limit price
            stop_price: New stop price
            quantity: New quantity
            time_in_force: New time in force
            
        Returns:
            Modification response
        """
        self.ensure_authenticated()
        endpoint = "/order/modifyorder"
        
        payload = {
            "orderId": order_id
        }
        
        # Add optional parameters if provided
        if price is not None:
            payload["price"] = price
        
        if stop_price is not None:
            payload["stopPrice"] = stop_price
        
        if quantity is not None:
            payload["orderQty"] = quantity
        
        if time_in_force is not None:
            payload["timeInForce"] = time_in_force
        
        return self._make_request("POST", endpoint, payload)
    
    def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Cancellation response
        """
        self.ensure_authenticated()
        endpoint = "/order/cancelorder"
        
        payload = {
            "orderId": order_id
        }
        
        return self._make_request("POST", endpoint, payload)
    
    def get_account_risk_status(self, account_id: int) -> Dict[str, Any]:
        """
        Get risk status for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Risk status information
        """
        self.ensure_authenticated()
        endpoint = f"/accountRiskStatus/item?accountId={account_id}"
        return self._make_request("GET", endpoint)
    
    def get_fills(self, account_id: int) -> List[Dict[str, Any]]:
        """
        Get fills for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            List of fills
        """
        self.ensure_authenticated()
        endpoint = f"/fill/list?accountId={account_id}"
        return self._make_request("GET", endpoint)
    
    def get_cash_balance(self, account_id: int) -> Dict[str, Any]:
        """
        Get cash balance for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Cash balance information
        """
        self.ensure_authenticated()
        endpoint = f"/cashBalance/item?accountId={account_id}"
        return self._make_request("GET", endpoint)
    
    def connect_websocket(self, callback: callable) -> None:
        """
        Connect to the Tradeovate WebSocket for real-time updates.
        
        Args:
            callback: Callback function for WebSocket messages
        """
        self.ensure_authenticated()
        
        # Close existing connection if any
        if self.ws_connected:
            self.disconnect_websocket()
        
        # Set up WebSocket connection
        def on_message(ws, message):
            try:
                data = json.loads(message)
                callback(data)
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {str(error)}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
            self.ws_connected = False
        
        def on_open(ws):
            logger.info("WebSocket connection established")
            self.ws_connected = True
            
            # Authenticate WebSocket
            auth_message = {
                "auth": {
                    "accessToken": self.access_token
                }
            }
            ws.send(json.dumps(auth_message))
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start WebSocket in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def disconnect_websocket(self) -> None:
        """Disconnect from the WebSocket."""
        if self.ws:
            self.ws.close()
            self.ws = None
            self.ws_connected = False
            logger.info("WebSocket disconnected")
    
    def subscribe_market_data(self, 
                             symbol: str, 
                             callback: callable,
                             chart_type: str = "Tick",
                             timeframe: str = "1 min") -> None:
        """
        Subscribe to market data for a symbol.
        
        Args:
            symbol: Trading symbol
            callback: Callback function for market data
            chart_type: Chart type ("Tick", "DOM", "Chart")
            timeframe: Timeframe for chart data
        """
        self.ensure_authenticated()
        
        # Connect to market data WebSocket if not already connected
        if not self.md_ws_connected:
            self._connect_md_websocket()
        
        # Register callback for this symbol
        self.md_ws_callbacks[symbol] = callback
        
        # Send subscription request
        subscription = {
            "chart": {
                "symbol": symbol,
                "chartType": chart_type,
                "timeframe": timeframe
            }
        }
        
        if self.md_ws and self.md_ws_connected:
            self.md_ws.send(json.dumps(subscription))
            logger.info(f"Subscribed to market data for {symbol}")
        else:
            logger.error("Cannot subscribe to market data: WebSocket not connected")
    
    def _connect_md_websocket(self) -> None:
        """Connect to the market data WebSocket."""
        # Close existing connection if any
        if self.md_ws_connected:
            self._disconnect_md_websocket()
        
        # Set up WebSocket connection
        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                # Check if this is a market data message
                if "chart" in data and "symbol" in data["chart"]:
                    symbol = data["chart"]["symbol"]
                    if symbol in self.md_ws_callbacks:
                        self.md_ws_callbacks[symbol](data)
            except Exception as e:
                logger.error(f"Error processing market data WebSocket message: {str(e)}")
        
        def on_error(ws, error):
            logger.error(f"Market data WebSocket error: {str(error)}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"Market data WebSocket connection closed: {close_status_code} - {close_msg}")
            self.md_ws_connected = False
        
        def on_open(ws):
            logger.info("Market data WebSocket connection established")
            self.md_ws_connected = True
            
            # Authenticate WebSocket
            auth_message = {
                "auth": {
                    "accessToken": self.md_access_token
                }
            }
            ws.send(json.dumps(auth_message))
        
        # Create WebSocket connection
        self.md_ws = websocket.WebSocketApp(
            self.md_ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start WebSocket in a separate thread
        self.md_ws_thread = threading.Thread(target=self.md_ws.run_forever)
        self.md_ws_thread.daemon = True
        self.md_ws_thread.start()
    
    def _disconnect_md_websocket(self) -> None:
        """Disconnect from the market data WebSocket."""
        if self.md_ws:
            self.md_ws.close()
            self.md_ws = None
            self.md_ws_connected = False
            logger.info("Market data WebSocket disconnected")
    
    def _make_request(self, 
                     method: str, 
                     endpoint: str, 
                     data: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Make an HTTP request to the Tradeovate API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            
        Returns:
            API response
            
        Raises:
            TradeovateApiError: If the request fails
        """
        url = self.base_url + endpoint
        
        # Add authentication header if we have a token
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.access_token and not endpoint.startswith("/auth/accesstoken"):
            headers["Authorization"] = f"Bearer {self.access_token}"
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = self.session.post(url, headers=headers, json=data)
                else:
                    raise TradeovateApiError(f"Unsupported HTTP method: {method}")
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse JSON response
                result = response.json()
                
                # Check for API errors
                if isinstance(result, dict) and "errorText" in result:
                    raise TradeovateApiError(
                        f"API error: {result.get('errorText')}",
                        status_code=response.status_code,
                        response=result
                    )
                
                return result
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise TradeovateApiError(f"Request failed after {self.max_retries} attempts: {str(e)}")
            
            except json.JSONDecodeError:
                raise TradeovateApiError(f"Invalid JSON response from API")
    
    def close(self) -> None:
        """Close all connections and clean up resources."""
        self.disconnect_websocket()
        self._disconnect_md_websocket()
        self.session.close()
        logger.info("Tradeovate API client closed")
