from typing import Dict, List
import numpy as np
from datamodel import OrderDepth, TradingState, Order


class Trader:
    def __init__(self):
        # Position limits for each product
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }
        
        # Track our current positions
        self.positions = {product: 0 for product in self.position_limits}
        
        # Price history for each product
        self.price_history = {product: [] for product in self.position_limits}
        
        # Moving average window sizes
        self.short_window = 5
        self.long_window = 20
        
        # For mean reversion calculation on SQUID_INK
        self.squid_avg_window = 10
        
        # Fair value for stable RAINFOREST_RESIN (to be adjusted based on market data)
        self.rainforest_fair_value = 10000  # Initial value, will be refined
        
        # Volatility tracker for SQUID_INK
        self.squid_volatility = 0
        
        # Trade history for each product
        self.trade_history = {product: [] for product in self.position_limits}

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Main trading logic - processes market data and generates orders
        """
        # Initialize the result dict
        result = {}
        
        # Update our internal position tracking
        for product, position in state.position.items():
            self.positions[product] = position
        
        # Update trade history from the state
        for product, trades in state.own_trades.items():
            for trade in trades:
                if trade.timestamp == state.timestamp:  # Only add new trades
                    self.trade_history[product].append(trade)
        
        # Update market data and calculate mid prices
        for product, order_depth in state.order_depths.items():
            mid_price = self.calculate_mid_price(order_depth)
            if mid_price is not None:
                self.price_history[product].append(mid_price)
        
        # Iterate over all the available products
        for product in state.order_depths.keys():
            # Initialize the list of orders for this product
            orders: List[Order] = []
            
            # Apply appropriate trading strategy based on product
            if product == 'RAINFOREST_RESIN':
                orders = self.trade_rainforest_resin(state, product)
            elif product == 'KELP':
                orders = self.trade_kelp(state, product)
            elif product == 'SQUID_INK':
                orders = self.trade_squid_ink(state, product)
            
            # Add the orders to the result dict if there are any
            if orders:
                result[product] = orders
            
        return result

    def trade_rainforest_resin(self, state: TradingState, product: str) -> List[Order]:
        """
        Trading strategy for RAINFOREST_RESIN - stable value market making
        """
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        
        # Refine our fair value estimate if we have enough price history
        if len(self.price_history[product]) > 30:
            # Update fair value based on recent average
            recent_prices = self.price_history[product][-30:]
            self.rainforest_fair_value = sum(recent_prices) / len(recent_prices)
        
        # Define the fair value for RAINFOREST_RESIN
        fair_value = self.rainforest_fair_value
        
        # Current position and limits
        position = self.positions.get(product, 0)
        position_limit = self.position_limits[product]
        
        # Calculate available position capacity
        buy_capacity = position_limit - position
        sell_capacity = position_limit + position
        
        # Market making: buy below fair value, sell above fair value
        if len(order_depth.sell_orders) > 0:
            # Sort sell orders by price (ascending)
            sell_prices = sorted(order_depth.sell_orders.keys())
            for ask_price in sell_prices:
                # Only buy if the price is below fair value
                if ask_price < fair_value:
                    # Calculate available volume to buy
                    available_volume = abs(order_depth.sell_orders[ask_price])
                    # Limit to our capacity
                    volume_to_buy = min(available_volume, buy_capacity)
                    
                    if volume_to_buy > 0:
                        orders.append(Order(product, ask_price, volume_to_buy))
                        buy_capacity -= volume_to_buy
                        
                        # Exit if we've reached position limit
                        if buy_capacity <= 0:
                            break
        
        if len(order_depth.buy_orders) > 0:
            # Sort buy orders by price (descending)
            buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            for bid_price in buy_prices:
                # Only sell if the price is above fair value
                if bid_price > fair_value:
                    # Calculate available volume to sell
                    available_volume = order_depth.buy_orders[bid_price]
                    # Limit to our capacity
                    volume_to_sell = min(available_volume, sell_capacity)
                    
                    if volume_to_sell > 0:
                        orders.append(Order(product, bid_price, -volume_to_sell))
                        sell_capacity -= volume_to_sell
                        
                        # Exit if we've reached position limit
                        if sell_capacity <= 0:
                            break
        
        return orders

    def trade_kelp(self, state: TradingState, product: str) -> List[Order]:
        """
        Trading strategy for KELP - trend following since it goes up and down over time
        """
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        
        # Current position and limits
        position = self.positions.get(product, 0)
        position_limit = self.position_limits[product]
        
        # Calculate available position capacity
        buy_capacity = position_limit - position
        sell_capacity = position_limit + position
        
        # Calculate trend signals
        trend_signal = self.calculate_trend_signal(product)
        
        # Calculate mid price if possible
        mid_price = self.calculate_mid_price(order_depth)
        if mid_price is None and len(self.price_history[product]) > 0:
            # Use the last known price if current mid price is not available
            mid_price = self.price_history[product][-1]
        elif mid_price is None:
            # Skip if we don't have any price information
            return []
        
        # Base fair value on current mid price
        fair_value = mid_price
        
        # Trade based on detected trend
        if trend_signal > 0:  # Uptrend
            # In an uptrend, we want to buy
            if buy_capacity > 0 and len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                volume_to_buy = min(abs(order_depth.sell_orders[best_ask]), buy_capacity)
                if volume_to_buy > 0:
                    orders.append(Order(product, best_ask, volume_to_buy))
        
        elif trend_signal < 0:  # Downtrend
            # In a downtrend, we want to sell
            if sell_capacity > 0 and len(order_depth.buy_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                volume_to_sell = min(order_depth.buy_orders[best_bid], sell_capacity)
                if volume_to_sell > 0:
                    orders.append(Order(product, best_bid, -volume_to_sell))
        
        else:  # No clear trend - do market making
            # Buy orders (if price is good)
            if len(order_depth.sell_orders) > 0 and buy_capacity > 0:
                best_ask = min(order_depth.sell_orders.keys())
                if best_ask < fair_value:  # Only buy below fair value
                    volume_to_buy = min(abs(order_depth.sell_orders[best_ask]), buy_capacity)
                    if volume_to_buy > 0:
                        orders.append(Order(product, best_ask, volume_to_buy))
            
            # Sell orders (if price is good)
            if len(order_depth.buy_orders) > 0 and sell_capacity > 0:
                best_bid = max(order_depth.buy_orders.keys())
                if best_bid > fair_value:  # Only sell above fair value
                    volume_to_sell = min(order_depth.buy_orders[best_bid], sell_capacity)
                    if volume_to_sell > 0:
                        orders.append(Order(product, best_bid, -volume_to_sell))
        
        return orders

    def trade_squid_ink(self, state: TradingState, product: str) -> List[Order]:
        """
        Trading strategy for SQUID_INK - focus on mean reversion as it has volatile prices
        that tend to revert after large swings
        """
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        
        # Current position and limits
        position = self.positions.get(product, 0)
        position_limit = self.position_limits[product]
        
        # Calculate available position capacity
        buy_capacity = position_limit - position
        sell_capacity = position_limit + position
        
        # Need some price history for mean reversion strategy
        if len(self.price_history[product]) < self.squid_avg_window:
            # Not enough history, use simple market making
            return self.simple_market_making(state, product)
        
        # Calculate current price and recent average
        current_price = self.price_history[product][-1]
        recent_avg = sum(self.price_history[product][-self.squid_avg_window:]) / self.squid_avg_window
        
        # Calculate standard deviation to measure volatility
        if len(self.price_history[product]) >= self.squid_avg_window:
            recent_prices = self.price_history[product][-self.squid_avg_window:]
            self.squid_volatility = np.std(recent_prices)
        
        # Calculate the deviation from the mean in terms of standard deviations (z-score)
        if self.squid_volatility > 0:
            z_score = (current_price - recent_avg) / self.squid_volatility
        else:
            z_score = 0
        
        # Thresholds for trading decisions
        buy_threshold = -1.0  # Buy when price is 1 std dev below mean
        sell_threshold = 1.0  # Sell when price is 1 std dev above mean
        
        # Mean reversion trading logic
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            
            # If price is significantly above average, sell
            if z_score > sell_threshold and sell_capacity > 0:
                volume_to_sell = min(order_depth.buy_orders[best_bid], sell_capacity)
                if volume_to_sell > 0:
                    orders.append(Order(product, best_bid, -volume_to_sell))
            
            # If price is significantly below average, buy
            elif z_score < buy_threshold and buy_capacity > 0:
                volume_to_buy = min(abs(order_depth.sell_orders[best_ask]), buy_capacity)
                if volume_to_buy > 0:
                    orders.append(Order(product, best_ask, volume_to_buy))
            
            # If price is near the average, do more conservative market making
            else:
                # Use a tighter fair value (closer to recent average)
                fair_value = recent_avg
                
                # Buy only if price is below fair value
                if best_ask < fair_value and buy_capacity > 0:
                    volume_to_buy = min(abs(order_depth.sell_orders[best_ask]), buy_capacity)
                    if volume_to_buy > 0:
                        orders.append(Order(product, best_ask, volume_to_buy))
                
                # Sell only if price is above fair value
                if best_bid > fair_value and sell_capacity > 0:
                    volume_to_sell = min(order_depth.buy_orders[best_bid], sell_capacity)
                    if volume_to_sell > 0:
                        orders.append(Order(product, best_bid, -volume_to_sell))
        
        return orders

    def simple_market_making(self, state: TradingState, product: str) -> List[Order]:
        """
        Simple market making strategy as a fallback
        """
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        
        # Current position and limits
        position = self.positions.get(product, 0)
        position_limit = self.position_limits[product]
        
        # Calculate available position capacity
        buy_capacity = position_limit - position
        sell_capacity = position_limit + position
        
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            # Get best bid and ask
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            
            # Calculate mid price as fair value
            fair_value = (best_bid + best_ask) / 2
            
            # Buy if price is below fair value
            if best_ask < fair_value and buy_capacity > 0:
                volume_to_buy = min(abs(order_depth.sell_orders[best_ask]), buy_capacity)
                if volume_to_buy > 0:
                    orders.append(Order(product, best_ask, volume_to_buy))
            
            # Sell if price is above fair value
            if best_bid > fair_value and sell_capacity > 0:
                volume_to_sell = min(order_depth.buy_orders[best_bid], sell_capacity)
                if volume_to_sell > 0:
                    orders.append(Order(product, best_bid, -volume_to_sell))
        
        return orders

    def calculate_mid_price(self, order_depth: OrderDepth):
        """Calculate the mid price from the order book"""
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return None

    def calculate_trend_signal(self, product: str):
        """
        Calculate trend signal based on moving averages
        Returns: 1 for uptrend, -1 for downtrend, 0 for no trend
        """
        history = self.price_history.get(product, [])
        
        # Not enough data for both short and long window
        if len(history) < self.long_window:
            return 0
            
        # Calculate short-term moving average
        short_ma = sum(history[-self.short_window:]) / self.short_window
        
        # Calculate long-term moving average
        long_ma = sum(history[-self.long_window:]) / self.long_window
        
        # Determine trend direction
        if short_ma > long_ma * 1.005:  # Add small threshold to avoid noise
            return 1  # Uptrend
        elif short_ma < long_ma * 0.995:  # Add small threshold to avoid noise
            return -1  # Downtrend
        else:
            return 0  # No clear trend