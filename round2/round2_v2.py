from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
import math

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET_1 = "PICNIC_BASKET_1"
    PICNIC_BASKET_2 = "PICNIC_BASKET_2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBE = "DJEMBE"

class AdaptiveMarketMaker:
    def __init__(self, initial_edge: float = 2.0, min_edge: float = 1.0, max_edge: float = 5.0,
                 execution_threshold: float = 0.3, adjustment_window: int = 20,
                 adjustment_step: float = 0.5):
        """
        Initialize the adaptive market maker.
        
        Args:
            initial_edge: Initial edge from fair price
            min_edge: Minimum allowed edge
            max_edge: Maximum allowed edge
            execution_threshold: Minimum execution rate to maintain current edge
            adjustment_window: Number of ticks to consider for execution rate
            adjustment_step: How much to adjust edge by
        """
        self.initial_edge = initial_edge
        self.current_edge = initial_edge
        self.min_edge = min_edge
        self.max_edge = max_edge
        self.execution_threshold = execution_threshold
        self.adjustment_window = adjustment_window
        self.adjustment_step = adjustment_step
        
        # Track fill history
        self.fill_history = []
        self.last_adjustment_tick = 0
        
    def place_quotes(self, fair_price: float) -> tuple[float, float]:
        """
        Generate bid and ask quotes based on current edge.
        
        Args:
            fair_price: Current fair price
            
        Returns:
            tuple: (bid_price, ask_price)
        """
        bid_price = math.floor(fair_price - self.current_edge)
        ask_price = math.ceil(fair_price + self.current_edge)
        return bid_price, ask_price
        
    def record_fill(self, successful: bool, current_tick: int) -> None:
        """
        Record whether a quote was filled.
        
        Args:
            successful: Whether the quote was filled
            current_tick: Current timestamp/tick
        """
        self.fill_history.append((current_tick, successful))
        
        # Maintain window size
        while len(self.fill_history) > self.adjustment_window:
            self.fill_history.pop(0)
            
        # Adjust edge if needed
        if current_tick - self.last_adjustment_tick >= self.adjustment_window:
            self.adjust_edge(current_tick)
            
    def adjust_edge(self, current_tick: int) -> None:
        """
        Adjust quoting edge based on recent execution rate.
        
        Args:
            current_tick: Current timestamp/tick
        """
        if not self.fill_history:
            return
            
        # Calculate execution rate
        successful_fills = sum(1 for _, success in self.fill_history if success)
        execution_rate = successful_fills / len(self.fill_history)
        
        # Adjust edge based on execution rate
        if execution_rate < self.execution_threshold:
            # Reduce edge if execution rate is too low
            self.current_edge = max(self.min_edge, self.current_edge - self.adjustment_step)
        elif execution_rate > 0.8:  # High execution rate
            # Increase edge if execution rate is high
            self.current_edge = min(self.max_edge, self.current_edge + self.adjustment_step)
            
        self.last_adjustment_tick = current_tick
        
    def reset(self) -> None:
        """Reset the market maker to initial state."""
        self.current_edge = self.initial_edge
        self.fill_history = []
        self.last_adjustment_tick = 0

class Trader:
    def __init__(self):
        self.kelp_prices = []
        self.kelp_vwap = []
        self.kelp_mmmid = []
        self.ink_prices = []  # Added for SQUID_INK price tracking
        
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBE: 60,
            Product.PICNIC_BASKET_1: 60,
            Product.PICNIC_BASKET_2: 100
        }
        
        # Initialize position tracking
        self.positions = {
            Product.RAINFOREST_RESIN: 0,
            Product.KELP: 0,
            Product.SQUID_INK: 0,
            Product.CROISSANTS: 0,
            Product.JAMS: 0,
            Product.DJEMBE: 0,
            Product.PICNIC_BASKET_1: 0,
            Product.PICNIC_BASKET_2: 0
        }

    def calc_mid(self, order_depth: OrderDepth) -> float:
        """Calculate mid price from order depth"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0.0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def estimate_fair_price(self, order_depth: OrderDepth, volume_threshold: int = 10) -> float:
        """
        Estimate the fair price based on large volume levels in the order book.
        
        Args:
            order_depth: OrderDepth object containing buy and sell orders
            volume_threshold: Minimum volume required to consider a price level
            
        Returns:
            float: Estimated fair price, or 0.0 if insufficient data
        """
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0.0
            
        # Find highest bid with sufficient volume
        valid_bids = [price for price, volume in order_depth.buy_orders.items() 
                     if volume >= volume_threshold]
        if not valid_bids:
            return 0.0
        highest_bid = max(valid_bids)
        
        # Find lowest ask with sufficient volume
        valid_asks = [price for price, volume in order_depth.sell_orders.items() 
                     if abs(volume) >= volume_threshold]
        if not valid_asks:
            return 0.0
        lowest_ask = min(valid_asks)
        
        # Calculate fair price as midpoint
        return (highest_bid + lowest_ask) / 2

    def generate_trading_signal(self, spread_series: List[float], n: int = 20, m: int = 5, threshold: float = 1.0) -> str:
        """
        Generate trading signals based on modified z-score logic.
        
        Args:
            spread_series: List of spread prices (basket price - synthetic price)
            n: Rolling window size for mean calculation
            m: Rolling window size for standard deviation calculation
            threshold: Z-score threshold for generating signals
            
        Returns:
            str: 'long', 'short', or 'hold'
        """
        if len(spread_series) < max(n, m):
            return 'hold'
            
        # Calculate rolling mean using window size n
        rolling_mean = np.mean(spread_series[-n:])
        
        # Calculate rolling standard deviation using window size m
        rolling_std = np.std(spread_series[-m:])
        
        # Handle case where standard deviation is zero
        if rolling_std == 0:
            return 'hold'
            
        # Calculate z-score
        current_spread = spread_series[-1]
        z_score = (current_spread - rolling_mean) / rolling_std
        
        # Generate signal
        if z_score < -threshold:
            return 'long'
        elif z_score > threshold:
            return 'short'
        else:
            return 'hold'

    def get_position(self, product: str, state: TradingState) -> int:
        return state.position.get(product, 0)

    # Returns buy_order_volume, sell_order_volume
    def take_best_orders(self, product: str, fair_value: int, take_width:float, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int, prevent_adverse: bool = False, adverse_volume: int = 0) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if prevent_adverse:
                if best_ask_amount <= adverse_volume and best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position) # max amt to buy 
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity)) 
                        buy_order_volume += quantity
            else:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position) # max amt to buy 
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity)) 
                        buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if prevent_adverse:
                if (best_bid >= fair_value + take_width) and (best_bid_amount <= adverse_volume):
                    quantity = min(best_bid_amount, position_limit + position) # should be the max we can sell 
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity

            else:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position) # should be the max we can sell 
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity

        return buy_order_volume, sell_order_volume
    
    def market_make(self, product: str, orders: List[Order], bid: int, ask: int, position: int, buy_order_volume: int, sell_order_volume: int) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))  # Sell order
    
        
        return buy_order_volume, sell_order_volume
    
    def clear_position_order(self, product: str, fair_value: float, width: int, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int) -> List[Order]:
        
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = round(fair_value)
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        # fair_for_ask = fair_for_bid = fair

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                # clear_quantity = position_after_take
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                # clear_quantity = abs(position_after_take)
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, method = "mid_price", min_vol = 0) -> float:
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return mid_price
        elif method == "mid_price_with_vol_filter":
            if len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]) ==0 or len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]) ==0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2
                return mid_price
            else:   
                best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                mid_price = (best_ask + best_bid) / 2
            return mid_price

    def resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        
        # Calculate ask and bid prices with fallback
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(aaf) if aaf else fair_value + 2
        bbbf = max(bbf) if bbf else fair_value - 2
        
        # Take Orders
        buy_order_volume, sell_order_volume = self.take_best_orders(Product.RAINFOREST_RESIN, fair_value, 0.5, orders, order_depth, position, buy_order_volume, sell_order_volume)
        # Clear Position Orders
        buy_order_volume, sell_order_volume = self.clear_position_order(Product.RAINFOREST_RESIN, fair_value, 1, orders, order_depth, position, buy_order_volume, sell_order_volume)
        # Market Make
        buy_order_volume, sell_order_volume = self.market_make(Product.RAINFOREST_RESIN, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)

        return orders
    

    def kelp_orders(self, order_depth: OrderDepth, timespan:int, width: float, kelp_take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:    
            
            # Calculate Fair
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            mmmid_price = (mm_ask + mm_bid) / 2    
            self.kelp_prices.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            if volume != 0:
                vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
                self.kelp_vwap.append({"vol": volume, "vwap": vwap})
            
            if len(self.kelp_vwap) > timespan:
                self.kelp_vwap.pop(0)
            
            if len(self.kelp_prices) > timespan:
                self.kelp_prices.pop(0)
        
            # fair_value = sum([x["vwap"]*x['vol'] for x in self.kelp_vwap]) / sum([x['vol'] for x in self.kelp_vwap])=
            # fair_value = sum(self.kelp_prices) / len(self.kelp_prices)
            fair_value = mmmid_price

            # only taking best bid/ask
            buy_order_volume, sell_order_volume = self.take_best_orders(Product.KELP, fair_value, kelp_take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, True, 20)
            
            # Clear Position Orders
            buy_order_volume, sell_order_volume = self.clear_position_order(Product.KELP, fair_value, 2, orders, order_depth, position, buy_order_volume, sell_order_volume)
            
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            # Market Make
            buy_order_volume, sell_order_volume = self.market_make(Product.KELP, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)

        return orders
    
    def squid_ink_orders(self, order_depth: OrderDepth, fair_value: int, width: float, position: int, position_limit: int, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        product = Product.SQUID_INK
        current_pos = self.get_position(product, state)
        orders_placed = 0
        MAX_ORDERS = 50
        
        # Calculate fair value with KELP correlation
        kelp_mid = 0
        if Product.KELP in state.order_depths:
            kelp_depth = state.order_depths[Product.KELP]
            if kelp_depth.buy_orders and kelp_depth.sell_orders:
                kelp_best_bid = max(kelp_depth.buy_orders.keys())
                kelp_best_ask = min(kelp_depth.sell_orders.keys())
                kelp_mid = (kelp_best_bid + kelp_best_ask) / 2
        
        hour = (state.timestamp // 10000) % 24
        time_factor = 1.003 if 14 <= hour < 18 else 0.997  # Late afternoon boost
        
        ink_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        ink_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
        base_mid = (ink_bid + ink_ask) / 2 if ink_bid and ink_ask else self.ink_prices[-1] if self.ink_prices else 1965
        
        # Incorporate KELP momentum (0.82 correlation)
        kelp_ma5 = np.mean(self.kelp_prices[-5:]) if len(self.kelp_prices) >= 5 else kelp_mid
        kelp_change = kelp_mid - kelp_ma5 if kelp_mid else 0
        
        # Volatility adjustment (20-period STD)
        volatility = np.std(self.ink_prices[-20:]) if len(self.ink_prices) >= 20 else 2.5
        
        fair_value = base_mid * (1 + 0.82 * kelp_change/1000) * time_factor + volatility * 0.3
        self.ink_prices.append(fair_value)
        
        # Market making logic
        recent_vol = np.std(self.ink_prices[-10:]) if len(self.ink_prices) >= 10 else 3
        base_spread = max(3, min(6, recent_vol * 1.5))
        position_penalty = abs(current_pos)/self.LIMIT[product] * 2
        spread = base_spread * (1 + position_penalty)
        
        # Trend following spread adjustment
        kelp_trend = np.mean(self.kelp_prices[-3:]) - np.mean(self.kelp_prices[-10:]) if len(self.kelp_prices) >= 10 else 0
        spread -= abs(kelp_trend) * 0.1
        
        bid_price = math.floor(fair_value - spread/2)
        ask_price = math.ceil(fair_value + spread/2)
        
        # Prevent bid/ask overlap
        if bid_price >= ask_price:
            ask_price = bid_price + 1
        
        # Order sizing with inventory management - ensure we don't exceed limits
        max_buy = min(10, self.LIMIT[product] - current_pos)  # Reduced from 15 to 10
        max_sell = min(10, self.LIMIT[product] + current_pos)  # Reduced from 15 to 10
        
        # Add market making orders
        if orders_placed < MAX_ORDERS and max_buy > 0:
            orders.append(Order(product, bid_price, max_buy))
            orders_placed += 1
        if orders_placed < MAX_ORDERS and max_sell > 0:
            orders.append(Order(product, ask_price, -max_sell))
            orders_placed += 1
        
        # Liquidity taking (mean reversion) - with aggregation by price level
        ma_20 = np.mean(self.ink_prices[-20:]) if len(self.ink_prices) >= 20 else fair_value
        
        # Aggregate sell orders
        sell_aggregate = {}
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask <= fair_value - 2 or ask <= ma_20 * 0.995:
                volume = min(-vol, self.LIMIT[product] - current_pos, 5)
                if volume > 0:
                    if ask in sell_aggregate:
                        sell_aggregate[ask] += volume
                    else:
                        sell_aggregate[ask] = volume
        
        # Add aggregated sell orders
        for ask, volume in sorted(sell_aggregate.items()):
            if orders_placed >= MAX_ORDERS:
                break
            orders.append(Order(product, ask, volume))
            orders_placed += 1
        
        # Aggregate buy orders
        buy_aggregate = {}
        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid >= fair_value + 2 or bid >= ma_20 * 1.005:
                volume = max(-vol, -self.LIMIT[product] - current_pos, -5)
                if volume < 0:
                    if bid in buy_aggregate:
                        buy_aggregate[bid] += volume
                    else:
                        buy_aggregate[bid] = volume
        
        # Add aggregated buy orders
        for bid, volume in sorted(buy_aggregate.items(), reverse=True):
            if orders_placed >= MAX_ORDERS:
                break
            orders.append(Order(product, bid, volume))
            orders_placed += 1
        
        # Hard stop-loss at 2.5% deviation from 50-period MA - only if under order limit
        if orders_placed < MAX_ORDERS:
            ma_50 = np.mean(self.ink_prices[-50:]) if len(self.ink_prices) >= 50 else fair_value
            if current_pos > 0 and fair_value < ma_50 * 0.975:
                # Instead of overwriting orders, add stop-loss order
                orders.append(Order(product, bid_price, -current_pos))
                orders_placed += 1
            elif current_pos < 0 and fair_value > ma_50 * 1.025:
                # Instead of overwriting orders, add stop-loss order
                orders.append(Order(product, ask_price, -current_pos))
                orders_placed += 1
        
        # Defensive programming: Final safety check to ensure we never return more than MAX_ORDERS
        if len(orders) > MAX_ORDERS:
            orders = orders[:MAX_ORDERS]
            print(f"Warning: Orders exceeded limit of {MAX_ORDERS}, truncated to first {MAX_ORDERS} orders")
        
        return orders

    def basket_arbitrage(self, state: TradingState) -> dict:
        """Execute basket arbitrage strategy between baskets and their components"""
        result = {}
        window_size = 20
        z_threshold = 1.0
        
        # Initialize price history if not exists
        if not hasattr(self, 'basket_spreads'):
            self.basket_spreads = {
                Product.PICNIC_BASKET_1: [],
                Product.PICNIC_BASKET_2: []
            }
        
        # Calculate synthetic prices and spreads
        for basket, components in [
            (Product.PICNIC_BASKET_1, [(Product.CROISSANTS, 6), (Product.JAMS, 3), (Product.DJEMBE, 1)]),
            (Product.PICNIC_BASKET_2, [(Product.CROISSANTS, 4), (Product.JAMS, 2)])
        ]:
            if basket not in state.order_depths:
                continue
                
            # Calculate basket mid price
            basket_mid = self.calc_mid(state.order_depths[basket])
            if basket_mid == 0:
                continue
                
            # Calculate synthetic price
            synthetic_price = 0
            for component, quantity in components:
                if component not in state.order_depths:
                    break
                component_mid = self.calc_mid(state.order_depths[component])
                if component_mid == 0:
                    break
                synthetic_price += component_mid * quantity
            else:  # Only execute if all components have valid prices
                # Calculate spread
                spread = basket_mid - synthetic_price
                self.basket_spreads[basket].append(spread)
                
                # Maintain rolling window
                if len(self.basket_spreads[basket]) > window_size:
                    self.basket_spreads[basket].pop(0)
                
                # Calculate z-score if we have enough data
                if len(self.basket_spreads[basket]) >= window_size:
                    spread_mean = np.mean(self.basket_spreads[basket])
                    spread_std = np.std(self.basket_spreads[basket])
                    if spread_std > 0:  # Avoid division by zero
                        z_score = (spread - spread_mean) / spread_std
                        
                        # Execute arbitrage based on z-score
                        basket_position = self.get_position(basket, state)
                        basket_orders = []
                        
                        if z_score > z_threshold and basket_position < self.LIMIT[basket]:
                            # Sell basket, buy components
                            # First check if we have enough liquidity
                            if state.order_depths[basket].buy_orders:
                                best_bid = max(state.order_depths[basket].buy_orders.keys())
                                basket_volume = min(
                                    state.order_depths[basket].buy_orders[best_bid],
                                    self.LIMIT[basket] - basket_position
                                )
                                if basket_volume > 0:
                                    basket_orders.append(Order(basket, best_bid, -basket_volume))
                                    
                                    # Buy components
                                    for component, quantity in components:
                                        if component not in result:
                                            result[component] = []
                                        component_position = self.get_position(component, state)
                                        if state.order_depths[component].sell_orders:
                                            best_ask = min(state.order_depths[component].sell_orders.keys())
                                            component_volume = min(
                                                -state.order_depths[component].sell_orders[best_ask],
                                                self.LIMIT[component] - component_position,
                                                basket_volume * quantity
                                            )
                                            if component_volume > 0:
                                                result[component].append(Order(component, best_ask, component_volume))
                        
                        elif z_score < -z_threshold and basket_position > -self.LIMIT[basket]:
                            # Buy basket, sell components
                            if state.order_depths[basket].sell_orders:
                                best_ask = min(state.order_depths[basket].sell_orders.keys())
                                basket_volume = min(
                                    -state.order_depths[basket].sell_orders[best_ask],
                                    self.LIMIT[basket] + basket_position
                                )
                                if basket_volume > 0:
                                    basket_orders.append(Order(basket, best_ask, basket_volume))
                                    
                                    # Sell components
                                    for component, quantity in components:
                                        if component not in result:
                                            result[component] = []
                                        component_position = self.get_position(component, state)
                                        if state.order_depths[component].buy_orders:
                                            best_bid = max(state.order_depths[component].buy_orders.keys())
                                            component_volume = min(
                                                state.order_depths[component].buy_orders[best_bid],
                                                self.LIMIT[component] + component_position,
                                                basket_volume * quantity
                                            )
                                            if component_volume > 0:
                                                result[component].append(Order(component, best_bid, -component_volume))
                        
                        if basket_orders:
                            result[basket] = basket_orders
        
        return result

    def run(self, state: TradingState):
        result = {}
        
        # Load and validate previous state
        if state.traderData:
            try:
                traderData = jsonpickle.decode(state.traderData)
                
                # Validate and cleanse kelp_prices
                if "kelp_prices" in traderData:
                    self.kelp_prices = [float(x) for x in traderData["kelp_prices"] if isinstance(x, (int, float))]
                else:
                    self.kelp_prices = []
                
                # Validate and cleanse kelp_vwap
                if "kelp_vwap" in traderData:
                    self.kelp_vwap = [x for x in traderData["kelp_vwap"] 
                                    if isinstance(x, dict) and 
                                    "vol" in x and "vwap" in x and 
                                    isinstance(x["vol"], (int, float)) and 
                                    isinstance(x["vwap"], (int, float))]
                else:
                    self.kelp_vwap = []
                
                # Validate and cleanse ink_prices
                if "ink_prices" in traderData:
                    self.ink_prices = [float(x) for x in traderData["ink_prices"] if isinstance(x, (int, float))]
                else:
                    self.ink_prices = []
                
                # Validate and cleanse basket_spreads
                if "basket_spreads" in traderData:
                    self.basket_spreads = {
                        Product.PICNIC_BASKET_1: [float(x) for x in traderData["basket_spreads"][Product.PICNIC_BASKET_1] if isinstance(x, (int, float))],
                        Product.PICNIC_BASKET_2: [float(x) for x in traderData["basket_spreads"][Product.PICNIC_BASKET_2] if isinstance(x, (int, float))]
                    }
                else:
                    self.basket_spreads = {
                        Product.PICNIC_BASKET_1: [],
                        Product.PICNIC_BASKET_2: []
                    }
                
            except Exception as e:
                print(f"Error loading trader data: {e}")
                # Reset state on error
                self.kelp_prices = []
                self.kelp_vwap = []
                self.ink_prices = []
                self.basket_spreads = {
                    Product.PICNIC_BASKET_1: [],
                    Product.PICNIC_BASKET_2: []
                }

        # Trading parameters
        resin_fair_value = 10000  # Participant should calculate this value
        resin_width = 2
        resin_position_limit = 50

        kelp_make_width = 3.5
        kelp_take_width = 1
        kelp_position_limit = 50
        kelp_timespan = 10

        squid_position_limit = 50
        squid_make_width = 3.5
        squid_take_width = 1

        print(state.traderData)

        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position[Product.RAINFOREST_RESIN] if Product.RAINFOREST_RESIN in state.position else 0
            resin_orders = self.resin_orders(
                state.order_depths[Product.RAINFOREST_RESIN],
                resin_fair_value,
                resin_width,
                resin_position,
                resin_position_limit
            )
            result[Product.RAINFOREST_RESIN] = resin_orders

        if Product.KELP in state.order_depths:
            kelp_position = state.position[Product.KELP] if Product.KELP in state.position else 0
            kelp_orders = self.kelp_orders(
                state.order_depths[Product.KELP],
                kelp_timespan,
                kelp_make_width,
                kelp_take_width,
                kelp_position,
                kelp_position_limit
            )
            result[Product.KELP] = kelp_orders

        if Product.SQUID_INK in state.order_depths:
            squid_position = state.position[Product.SQUID_INK] if Product.SQUID_INK in state.position else 0
            squid_orders = self.squid_ink_orders(
                state.order_depths[Product.SQUID_INK],
                resin_fair_value,
                resin_width,
                squid_position,
                squid_position_limit,
                state
            )
            result[Product.SQUID_INK] = squid_orders

        # Execute basket arbitrage strategy
        basket_orders = self.basket_arbitrage(state)
        for product, orders in basket_orders.items():
            if product in result:
                result[product].extend(orders)
            else:
                result[product] = orders

        # Save state with validated data
        traderData = jsonpickle.encode({
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap,
            "ink_prices": self.ink_prices,
            "basket_spreads": self.basket_spreads
        })

        conversions = 1
        return result, conversions, traderData