from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Symbol
import math
import pandas as pd
import numpy as np
import statistics

class Trader:
    def __init__(self):
        self.position_limit = {"RAINFOREST_RESIN": 50, "KELP": 50}
        self.kelp_prices = []
        self.kelp_ema = None
        
    def calculate_fair_value(self, product: str, order_depth: OrderDepth) -> float:
        if product == "RAINFOREST_RESIN":
            return 10000  # Stable value from competition docs
            
        elif product == "KELP":
            # Calculate volume-weighted midprice
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            bid_volume = order_depth.buy_orders[best_bid]
            ask_volume = order_depth.sell_orders[best_ask]
            
            vwap = (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
            self.kelp_prices.append(vwap)
            
            # Use exponential moving average for prediction
            if len(self.kelp_prices) > 1:
                self.kelp_ema = 0.8 * vwap + 0.2 * self.starfruit_ema if self.starfruit_ema else vwap
                return self.kelp_ema
            return vwap

    def market_make(self, product: str, fair_price: float, position: int, order_depth: OrderDepth) -> List[Order]:
        orders = []
        spread = 3 if product == "RAINFOREST_RESIN" else 2
        position_limit = self.position_limit[product]
        
        # Calculate bid/ask prices with spread
        bid_price = math.floor(fair_price - spread)
        ask_price = math.ceil(fair_price + spread)
        
        # Adjust for position limits
        max_buy = position_limit - position
        max_sell = position_limit + position
        
        # Place buy orders
        if max_buy > 0:
            orders.append(Order(product, bid_price, max_buy))
            
        # Place sell orders
        if max_sell > 0:
            orders.append(Order(product, ask_price, -max_sell))
            
        return orders

    def take_liquidity(self, product: str, fair_price: float, order_depth: OrderDepth) -> List[Order]:
        orders = []
        
        # Take advantage of mispriced asks
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask < fair_price:
                volume = min(-vol, self.position_limit[product])
                orders.append(Order(product, ask, volume))
                
        # Take advantage of mispriced bids
        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > fair_price:
                volume = max(-vol, -self.position_limit[product])
                orders.append(Order(product, bid, volume))
                
        return orders

    def run(self, state: TradingState):
        result = {}
        
        for product in ["RAINFOREST_RESIN", "KELP"]:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                position = state.position.get(product, 0)
                
                # Calculate fair value
                fair_value = self.calculate_fair_value(product, order_depth)
                
                # Generate orders
                mm_orders = self.market_make(product, fair_value, position, order_depth)
                tl_orders = self.take_liquidity(product, fair_value, order_depth)
                
                # Combine and filter orders
                all_orders = mm_orders + tl_orders
                result[product] = [o for o in all_orders if o.quantity != 0]

        return result, None, ""
