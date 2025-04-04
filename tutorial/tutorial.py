from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

# Implement loader class? Then load into the strategies and then work on determining true values.

class Strategy:
	def __init__(self, symbol: str, pos_limit: int) -> None:
		self.symbol = symbol
		self.pos_limit = pos_limit
		
	def act(self, trading_state: TradingState) -> None:
		print('Not Implemented Yet')
		
	def run(self, trading_state: TradingState) -> list[Order]:
		self.orders = []
		self.act(trading_state)
		return self.orders
		
	def buy(self, price: int, quantity: int) -> None:
		self.orders.append(Order(self.symbol, price, quantity))
		
	def sell(self, price: int, quantity: int) -> None:
		self.orders.append(Order(self.symbol, price, -quantity))
		
	def save(self) -> JSON:
		return None
		
	def load(self, load_data: JSON) -> None:
		pass

class TradingStrategy(Strategy):
	def __init__(self, symbol: Symbol, pos_limit: int) -> None:
		super().__init__(symbol, pos_limit)
		self.deque = deque()
		self.size = 10
	def get_value(state: TradingState) -> int:
		print('Not Implemented')
	def act(self, state: TradingState) -> None:
		fair_value = self.get_value(state)
		# Finish implementing
	def save(self) -> JSON:
		return list(self.deque)
	def load(self, load_data: JSON) -> None:
		self.deque = deque(load_data)

class RainforestResinStrategy(TradingStrategy):
	def get_value(self, state: TradingState) -> int:
		return 0 # Stable, analyze this

class KelpStrategy(TradingStrategy):
	def get_value(self, state: TradingState) -> int:
		return 0 # Not stable, goes up and down over time

class Trader:
    def __init__(self) -> None:
	    limits = {
		    "RAINFOREST_RESIN": 50,
		    "KELP": 50,
	    }
	    self.strategies = {symbol: symbol_strat(symbol, limits[symbol]) for symbol, symbol_strat in {
		    "RAINFOREST_RESIN": RainforestResinStrategy,
		    "KELP": KelpStrategy,
	    }.items()}
	    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

				# Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData
