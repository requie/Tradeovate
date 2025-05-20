"""
Tradeovate API client for trading bot.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import aiohttp
import asyncio
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
        """Initialize the API client."""
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
        
        # HTTP session
        self.session = None
        
        # Base URLs based on environment
        self.base_url = self.LIVE_URL if use_live else self.DEMO_URL
        self.md_url = self.MD_URL
        self.ws_url = self.LIVE_WS_URL if use_live else self.DEMO_WS_URL
        self.md_ws_url = self.MD_WS_URL
        
        logger.info(f"Initialized Tradeovate API client for {'live' if use_live else 'demo'} environment")

    async def authenticate(self) -> Dict[str, Any]:
        """Authenticate with the API and obtain access tokens."""
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
            response = await self._make_request("POST", endpoint, payload)
            
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

    async def ensure_authenticated(self) -> None:
        """Ensure valid authentication tokens exist."""
        if (self.access_token is None or 
            self.token_expiry is None or 
            datetime.now(self.token_expiry.tzinfo) >= self.token_expiry - timedelta(seconds=self.token_refresh_margin)):
            
            logger.info("Authentication tokens missing or expiring soon, re-authenticating")
            await self.authenticate()

    async def get_user_info(self) -> Dict[str, Any]:
        """Get information about the authenticated user."""
        await self.ensure_authenticated()
        endpoint = "/auth/me"
        return await self._make_request("GET", endpoint)

    async def get_accounts(self) -> List[Dict[str, Any]]:
        """Get list of user accounts."""
        await self.ensure_authenticated()
        endpoint = "/account/list"
        return await self._make_request("GET", endpoint)

    async def get_account_info(self, account_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific account."""
        await self.ensure_authenticated()
        endpoint = f"/account/item?id={account_id}"
        return await self._make_request("GET", endpoint)

    async def get_positions(self, account_id: int) -> List[Dict[str, Any]]:
        """Get current positions for an account."""
        await self.ensure_authenticated()
        endpoint = f"/position/list?accountId={account_id}"
        return await self._make_request("GET", endpoint)

    async def get_orders(self, account_id: int, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get orders for an account."""
        await self.ensure_authenticated()
        endpoint = f"/order/list?accountId={account_id}"
        if status:
            endpoint += f"&status={status}"
        return await self._make_request("GET", endpoint)

    async def get_contract(self, contract_id: int) -> Dict[str, Any]:
        """Get contract details."""
        await self.ensure_authenticated()
        endpoint = f"/contract/item?id={contract_id}"
        return await self._make_request("GET", endpoint)

    async def find_contracts(self, name: str) -> List[Dict[str, Any]]:
        """Find contracts by name."""
        await self.ensure_authenticated()
        endpoint = f"/contract/find?name={name}"
        return await self._make_request("GET", endpoint)

    async def place_order(self, 
                         account_id: int,
                         contract_id: int,
                         action: str,
                         order_type: str,
                         quantity: int,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         time_in_force: str = "Day",
                         bracket_strategy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Place a new order."""
        await self.ensure_authenticated()
        endpoint = "/order/placeorder"
        
        payload = {
            "accountId": account_id,
            "contractId": contract_id,
            "action": action,
            "orderQty": quantity,
            "orderType": order_type,
            "timeInForce": time_in_force
        }
        
        if order_type in ["Limit", "StopLimit"] and price is not None:
            payload["price"] = price
        
        if order_type in ["Stop", "StopLimit"] and stop_price is not None:
            payload["stopPrice"] = stop_price
        
        if bracket_strategy:
            payload["bracketStrategy"] = bracket_strategy
        
        return await self._make_request("POST", endpoint, payload)

    async def modify_order(self, 
                          order_id: int,
                          price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          quantity: Optional[int] = None,
                          time_in_force: Optional[str] = None) -> Dict[str, Any]:
        """Modify an existing order."""
        await self.ensure_authenticated()
        endpoint = "/order/modifyorder"
        
        payload = {
            "orderId": order_id
        }
        
        if price is not None:
            payload["price"] = price
        
        if stop_price is not None:
            payload["stopPrice"] = stop_price
        
        if quantity is not None:
            payload["orderQty"] = quantity
        
        if time_in_force is not None:
            payload["timeInForce"] = time_in_force
        
        return await self._make_request("POST", endpoint, payload)

    async def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """Cancel an order."""
        await self.ensure_authenticated()
        endpoint = "/order/cancelorder"
        
        payload = {
            "orderId": order_id
        }
        
        return await self._make_request("POST", endpoint, payload)

    async def get_account_risk_status(self, account_id: int) -> Dict[str, Any]:
        """Get risk status for an account."""
        await self.ensure_authenticated()
        endpoint = f"/accountRiskStatus/item?accountId={account_id}"
        return await self._make_request("GET", endpoint)

    async def get_fills(self, account_id: int) -> List[Dict[str, Any]]:
        """Get fills for an account."""
        await self.ensure_authenticated()
        endpoint = f"/fill/list?accountId={account_id}"
        return await self._make_request("GET", endpoint)

    async def get_cash_balance(self, account_id: int) -> Dict[str, Any]:
        """Get cash balance for an account."""
        await self.ensure_authenticated()
        endpoint = f"/cashBalance/item?accountId={account_id}"
        return await self._make_request("GET", endpoint)

    async def _make_request(self, 
                          method: str, 
                          endpoint: str, 
                          data: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Make an HTTP request to the Tradeovate API."""
        url = self.base_url + endpoint
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.access_token and not endpoint.startswith("/auth/accesstoken"):
            headers["Authorization"] = f"Bearer {self.access_token}"
        
        for attempt in range(self.max_retries):
            try:
                if self.session is None:
                    self.session = aiohttp.ClientSession()
                
                if method.upper() == "GET":
                    async with self.session.get(url, headers=headers) as response:
                        response.raise_for_status()
                        result = await response.json()
                elif method.upper() == "POST":
                    async with self.session.post(url, headers=headers, json=data) as response:
                        response.raise_for_status()
                        result = await response.json()
                else:
                    raise TradeovateApiError(f"Unsupported HTTP method: {method}")
                
                if isinstance(result, dict) and "errorText" in result:
                    raise TradeovateApiError(
                        f"API error: {result.get('errorText')}",
                        status_code=response.status,
                        response=result
                    )
                
                return result
                
            except aiohttp.ClientError as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise TradeovateApiError(f"Request failed after {self.max_retries} attempts: {str(e)}")
            except json.JSONDecodeError:
                raise TradeovateApiError("Invalid JSON response from API")

    async def close(self) -> None:
        """Close all connections and clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
        
        if self.ws:
            self.ws.close()
            self.ws = None
            self.ws_connected = False
        
        if self.md_ws:
            self.md_ws.close()
            self.md_ws = None
            self.md_ws_connected = False
        
        logger.info("Tradeovate API client closed")