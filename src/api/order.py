"""
Order management module for Tradeovate API.
"""
from typing import Dict, Any, List, Optional, Union, Callable
import logging
import threading
import queue
import time
from datetime import datetime

from .client import TradeovateApiClient, TradeovateApiError

logger = logging.getLogger(__name__)

class Order:
    """
    Represents an order in the trading system.
    """
    
    # Order statuses
    STATUS_PENDING = "Pending"
    STATUS_WORKING = "Working"
    STATUS_COMPLETED = "Completed"
    STATUS_CANCELED = "Canceled"
    STATUS_REJECTED = "Rejected"
    
    # Order types
    TYPE_MARKET = "Market"
    TYPE_LIMIT = "Limit"
    TYPE_STOP = "Stop"
    TYPE_STOP_LIMIT = "StopLimit"
    
    # Order actions
    ACTION_BUY = "Buy"
    ACTION_SELL = "Sell"
    
    # Time in force options
    TIF_DAY = "Day"
    TIF_GTC = "GTC"  # Good Till Canceled
    TIF_IOC = "IOC"  # Immediate or Cancel
    TIF_FOK = "FOK"  # Fill or Kill
    
    def __init__(self, 
                 account_id: int,
                 contract_id: int,
                 action: str,
                 order_type: str,
                 quantity: int,
                 price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 time_in_force: str = "Day",
                 order_id: Optional[int] = None,
                 status: str = "Pending",
                 filled_quantity: int = 0,
                 average_fill_price: Optional[float] = None,
                 creation_time: Optional[datetime] = None,
                 last_update_time: Optional[datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an order.
        
        Args:
            account_id: Account ID
            contract_id: Contract ID
            action: Order action (Buy/Sell)
            order_type: Order type (Market/Limit/Stop/StopLimit)
            quantity: Order quantity
            price: Limit price (required for Limit and StopLimit orders)
            stop_price: Stop price (required for Stop and StopLimit orders)
            time_in_force: Time in force (Day/GTC/IOC/FOK)
            order_id: Order ID (None for new orders)
            status: Order status
            filled_quantity: Quantity filled
            average_fill_price: Average fill price
            creation_time: Order creation time
            last_update_time: Last update time
            metadata: Additional order metadata
        """
        self.account_id = account_id
        self.contract_id = contract_id
        self.action = action
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.order_id = order_id
        self.status = status
        self.filled_quantity = filled_quantity
        self.average_fill_price = average_fill_price
        self.creation_time = creation_time or datetime.now()
        self.last_update_time = last_update_time or self.creation_time
        self.metadata = metadata or {}
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_quantity == self.quantity
    
    @property
    def is_active(self) -> bool:
        """Check if order is active (pending or working)."""
        return self.status in [self.STATUS_PENDING, self.STATUS_WORKING]
    
    @property
    def remaining_quantity(self) -> int:
        """Get remaining quantity to be filled."""
        return self.quantity - self.filled_quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert order to dictionary for API requests.
        
        Returns:
            Order as dictionary
        """
        order_dict = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "action": self.action,
            "orderQty": self.quantity,
            "orderType": self.order_type,
            "timeInForce": self.time_in_force
        }
        
        # Add price for Limit and StopLimit orders
        if self.order_type in [self.TYPE_LIMIT, self.TYPE_STOP_LIMIT] and self.price is not None:
            order_dict["price"] = self.price
        
        # Add stop price for Stop and StopLimit orders
        if self.order_type in [self.TYPE_STOP, self.TYPE_STOP_LIMIT] and self.stop_price is not None:
            order_dict["stopPrice"] = self.stop_price
        
        return order_dict
    
    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'Order':
        """
        Create an order from API response.
        
        Args:
            response: API response
            
        Returns:
            Order object
        """
        # Map API response fields to Order constructor parameters
        order_id = response.get("id")
        account_id = response.get("accountId")
        contract_id = response.get("contractId")
        action = response.get("action")
        order_type = response.get("orderType")
        quantity = response.get("orderQty")
        price = response.get("price")
        stop_price = response.get("stopPrice")
        time_in_force = response.get("timeInForce")
        status = response.get("status")
        filled_quantity = response.get("filledQty", 0)
        average_fill_price = response.get("avgPrice")
        
        # Parse timestamps
        creation_time = None
        if "timestamp" in response:
            creation_time = datetime.fromtimestamp(response["timestamp"] / 1000)  # Convert ms to seconds
        
        last_update_time = None
        if "lastUpdateTime" in response:
            last_update_time = datetime.fromtimestamp(response["lastUpdateTime"] / 1000)
        
        return cls(
            account_id=account_id,
            contract_id=contract_id,
            action=action,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            order_id=order_id,
            status=status,
            filled_quantity=filled_quantity,
            average_fill_price=average_fill_price,
            creation_time=creation_time,
            last_update_time=last_update_time
        )
    
    def __str__(self) -> str:
        """String representation of the order."""
        return (f"Order(id={self.order_id}, {self.action} {self.quantity} @ "
                f"{self.price if self.price else 'Market'}, "
                f"status={self.status}, filled={self.filled_quantity})")

class OrderManager:
    """
    Manages orders for the trading system.
    
    This class handles order submission, modification, cancellation,
    and tracking of order status.
    """
    
    def __init__(self, api_client: TradeovateApiClient):
        """
        Initialize the order manager.
        
        Args:
            api_client: Tradeovate API client
        """
        self.api_client = api_client
        self.orders = {}  # order_id -> Order
        self.pending_orders = {}  # client_order_id -> Order
        self.order_callbacks = {}  # order_id -> callback
        
        # For handling order updates in a separate thread
        self.update_queue = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        
        # For generating client order IDs
        self.next_client_order_id = 1
        self.client_order_id_lock = threading.Lock()
    
    def start(self) -> None:
        """Start the order processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._process_update_queue)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Order processing thread started")
    
    def stop(self) -> None:
        """Stop the order processing thread."""
        self.is_processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            logger.info("Order processing thread stopped")
    
    def place_order(self, 
                   order: Order, 
                   callback: Optional[Callable[[Order], None]] = None) -> int:
        """
        Place a new order.
        
        Args:
            order: Order to place
            callback: Optional callback for order updates
            
        Returns:
            Client order ID for tracking
            
        Raises:
            TradeovateApiError: If order placement fails
        """
        # Generate client order ID
        with self.client_order_id_lock:
            client_order_id = self.next_client_order_id
            self.next_client_order_id += 1
        
        # Store callback if provided
        if callback:
            self.order_callbacks[client_order_id] = callback
        
        # Store order in pending orders
        self.pending_orders[client_order_id] = order
        
        try:
            # Submit order to API
            response = self.api_client.place_order(
                account_id=order.account_id,
                contract_id=order.contract_id,
                action=order.action,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force
            )
            
            # Update order with API response
            order_id = response.get("orderId")
            if order_id:
                order.order_id = order_id
                order.status = Order.STATUS_WORKING
                order.last_update_time = datetime.now()
                
                # Move from pending to active orders
                self.orders[order_id] = order
                del self.pending_orders[client_order_id]
                
                # Update callback reference
                if client_order_id in self.order_callbacks:
                    self.order_callbacks[order_id] = self.order_callbacks[client_order_id]
                    del self.order_callbacks[client_order_id]
                
                # Call callback with updated order
                if order_id in self.order_callbacks and self.order_callbacks[order_id]:
                    self.order_callbacks[order_id](order)
                
                logger.info(f"Order placed successfully: {order}")
            else:
                # Handle case where order ID is not returned
                logger.error(f"Order placement failed: No order ID returned")
                order.status = Order.STATUS_REJECTED
                
                # Call callback with rejected order
                if client_order_id in self.order_callbacks and self.order_callbacks[client_order_id]:
                    self.order_callbacks[client_order_id](order)
                
                # Clean up
                del self.pending_orders[client_order_id]
                if client_order_id in self.order_callbacks:
                    del self.order_callbacks[client_order_id]
                
                raise TradeovateApiError("Order placement failed: No order ID returned")
            
            return client_order_id
            
        except TradeovateApiError as e:
            # Handle API errors
            logger.error(f"Order placement failed: {str(e)}")
            
            # Update order status
            order.status = Order.STATUS_REJECTED
            order.metadata["rejection_reason"] = str(e)
            
            # Call callback with rejected order
            if client_order_id in self.order_callbacks and self.order_callbacks[client_order_id]:
                self.order_callbacks[client_order_id](order)
            
            # Clean up
            del self.pending_orders[client_order_id]
            if client_order_id in self.order_callbacks:
                del self.order_callbacks[client_order_id]
            
            # Re-raise exception
            raise
    
    def modify_order(self, 
                    order_id: int, 
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    quantity: Optional[int] = None,
                    time_in_force: Optional[str] = None) -> bool:
        """
        Modify an existing order.
        
        Args:
            order_id: Order ID
            price: New limit price
            stop_price: New stop price
            quantity: New quantity
            time_in_force: New time in force
            
        Returns:
            True if modification was successful
            
        Raises:
            TradeovateApiError: If order modification fails
        """
        # Check if order exists
        if order_id not in self.orders:
            logger.error(f"Cannot modify order {order_id}: Order not found")
            return False
        
        # Get order
        order = self.orders[order_id]
        
        # Check if order can be modified
        if not order.is_active:
            logger.error(f"Cannot modify order {order_id}: Order is not active")
            return False
        
        try:
            # Submit modification to API
            response = self.api_client.modify_order(
                order_id=order_id,
                price=price,
                stop_price=stop_price,
                quantity=quantity,
                time_in_force=time_in_force
            )
            
            # Update order with new values
            if price is not None:
                order.price = price
            
            if stop_price is not None:
                order.stop_price = stop_price
            
            if quantity is not None:
                order.quantity = quantity
            
            if time_in_force is not None:
                order.time_in_force = time_in_force
            
            order.last_update_time = datetime.now()
            
            # Call callback with updated order
            if order_id in self.order_callbacks and self.order_callbacks[order_id]:
                self.order_callbacks[order_id](order)
            
            logger.info(f"Order {order_id} modified successfully")
            return True
            
        except TradeovateApiError as e:
            # Handle API errors
            logger.error(f"Order modification failed: {str(e)}")
            
            # Call callback with current order
            if order_id in self.order_callbacks and self.order_callbacks[order_id]:
                self.order_callbacks[order_id](order)
            
            # Re-raise exception
            raise
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            True if cancellation was successful
            
        Raises:
            TradeovateApiError: If order cancellation fails
        """
        # Check if order exists
        if order_id not in self.orders:
            logger.error(f"Cannot cancel order {order_id}: Order not found")
            return False
        
        # Get order
        order = self.orders[order_id]
        
        # Check if order can be canceled
        if not order.is_active:
            logger.error(f"Cannot cancel order {order_id}: Order is not active")
            return False
        
        try:
            # Submit cancellation to API
            response = self.api_client.cancel_order(order_id=order_id)
            
            # Update order status
            order.status = Order.STATUS_CANCELED
            order.last_update_time = datetime.now()
            
            # Call callback with updated order
            if order_id in self.order_callbacks and self.order_callbacks[order_id]:
                self.order_callbacks[order_id](order)
            
            logger.info(f"Order {order_id} canceled successfully")
            return True
            
        except TradeovateApiError as e:
            # Handle API errors
            logger.error(f"Order cancellation failed: {str(e)}")
            
            # Call callback with current order
            if order_id in self.order_callbacks and self.order_callbacks[order_id]:
                self.order_callbacks[order_id](order)
            
            # Re-raise exception
            raise
    
    def cancel_all_orders(self, account_id: int) -> int:
        """
        Cancel all active orders for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Number of orders canceled
        """
        # Get active orders for account
        active_orders = [
            order_id for order_id, order in self.orders.items()
            if order.account_id == account_id and order.is_active
        ]
        
        # Cancel each order
        canceled_count = 0
        for order_id in active_orders:
            try:
                if self.cancel_order(order_id):
                    canceled_count += 1
            except Exception as e:
                logger.error(f"Error canceling order {order_id}: {str(e)}")
        
        logger.info(f"Canceled {canceled_count} orders for account {account_id}")
        return canceled_count
    
    def get_order(self, order_id: int) -> Optional[Order]:
        """
        Get an order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order or None if not found
        """
        return self.orders.get(order_id)
    
    def get_orders(self, 
                  account_id: Optional[int] = None, 
                  active_only: bool = False) -> List[Order]:
        """
        Get orders, optionally filtered by account and status.
        
        Args:
            account_id: Optional account ID filter
            active_only: Whether to return only active orders
            
        Returns:
            List of orders
        """
        orders = list(self.orders.values())
        
        # Filter by account ID if provided
        if account_id is not None:
            orders = [order for order in orders if order.account_id == account_id]
        
        # Filter by active status if requested
        if active_only:
            orders = [order for order in orders if order.is_active]
        
        return orders
    
    def sync_orders(self, account_id: int) -> None:
        """
        Synchronize orders with the API.
        
        This method fetches the latest order information from the API
        and updates the local order cache.
        
        Args:
            account_id: Account ID
        """
        try:
            # Fetch orders from API
            api_orders = self.api_client.get_orders(account_id=account_id)
            
            # Process each order
            for api_order in api_orders:
                order_id = api_order.get("id")
                if order_id:
                    if order_id in self.orders:
                        # Update existing order
                        self._update_order_from_api(self.orders[order_id], api_order)
                    else:
                        # Create new order
                        order = Order.from_api_response(api_order)
                        self.orders[order_id] = order
            
            logger.info(f"Synchronized {len(api_orders)} orders for account {account_id}")
            
        except Exception as e:
            logger.error(f"Error synchronizing orders: {str(e)}")
    
    def register_order_callback(self, 
                               order_id: int, 
                               callback: Callable[[Order], None]) -> None:
        """
        Register a callback for order updates.
        
        Args:
            order_id: Order ID
            callback: Callback function
        """
        self.order_callbacks[order_id] = callback
    
    def unregister_order_callback(self, order_id: int) -> None:
        """
        Unregister a callback for order updates.
        
        Args:
            order_id: Order ID
        """
        if order_id in self.order_callbacks:
            del self.order_callbacks[order_id]
    
    def handle_order_update(self, update: Dict[str, Any]) -> None:
        """
        Handle an order update from WebSocket.
        
        Args:
            update: Order update message
        """
        # Add to processing queue
        self.update_queue.put(update)
    
    def _process_update_queue(self) -> None:
        """Process order update queue in a separate thread."""
        while self.is_processing:
            try:
                # Get update from queue with timeout to allow checking is_processing
                try:
                    update = self.update_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the update
                self._process_order_update(update)
                
                # Mark task as done
                self.update_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing order update: {str(e)}")
    
    def _process_order_update(self, update: Dict[str, Any]) -> None:
        """
        Process an order update.
        
        Args:
            update: Order update message
        """
        try:
            # Extract order information
            if "order" in update:
                order_data = update["order"]
                order_id = order_data.get("id")
                
                if order_id:
                    if order_id in self.orders:
                        # Update existing order
                        order = self.orders[order_id]
                        self._update_order_from_api(order, order_data)
                        
                        # Call callback if registered
                        if order_id in self.order_callbacks and self.order_callbacks[order_id]:
                            self.order_callbacks[order_id](order)
                    else:
                        # Create new order
                        order = Order.from_api_response(order_data)
                        self.orders[order_id] = order
                        
                        # Call callback if registered
                        if order_id in self.order_callbacks and self.order_callbacks[order_id]:
                            self.order_callbacks[order_id](order)
            
            # Extract fill information
            elif "fill" in update:
                fill_data = update["fill"]
                order_id = fill_data.get("orderId")
                
                if order_id and order_id in self.orders:
                    # Update order with fill information
                    order = self.orders[order_id]
                    self._update_order_with_fill(order, fill_data)
                    
                    # Call callback if registered
                    if order_id in self.order_callbacks and self.order_callbacks[order_id]:
                        self.order_callbacks[order_id](order)
        
        except Exception as e:
            logger.error(f"Error processing order update: {str(e)}")
    
    def _update_order_from_api(self, order: Order, api_data: Dict[str, Any]) -> None:
        """
        Update an order with API data.
        
        Args:
            order: Order to update
            api_data: API data
        """
        # Update order fields
        if "status" in api_data:
            order.status = api_data["status"]
        
        if "filledQty" in api_data:
            order.filled_quantity = api_data["filledQty"]
        
        if "avgPrice" in api_data:
            order.average_fill_price = api_data["avgPrice"]
        
        if "lastUpdateTime" in api_data:
            order.last_update_time = datetime.fromtimestamp(api_data["lastUpdateTime"] / 1000)
        else:
            order.last_update_time = datetime.now()
    
    def _update_order_with_fill(self, order: Order, fill_data: Dict[str, Any]) -> None:
        """
        Update an order with fill information.
        
        Args:
            order: Order to update
            fill_data: Fill data
        """
        # Extract fill information
        fill_quantity = fill_data.get("qty", 0)
        fill_price = fill_data.get("price", 0.0)
        
        # Update order
        order.filled_quantity += fill_quantity
        
        # Calculate new average fill price
        if order.average_fill_price is None:
            order.average_fill_price = fill_price
        else:
            # Weighted average of previous fills and new fill
            previous_fill_value = (order.filled_quantity - fill_quantity) * order.average_fill_price
            new_fill_value = fill_quantity * fill_price
            order.average_fill_price = (previous_fill_value + new_fill_value) / order.filled_quantity
        
        # Update status if fully filled
        if order.filled_quantity >= order.quantity:
            order.status = Order.STATUS_COMPLETED
        
        order.last_update_time = datetime.now()
