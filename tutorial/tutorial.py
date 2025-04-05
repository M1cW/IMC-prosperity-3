from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import json # Convert to jsonpickle?
import numpy as np
import math

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

# Implement loader class? Then load into the strategies and then work on determining true values.

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 4000

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # truncate state.traderData, trader_data, and self.logs to same max length to fit log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
		observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()


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
		
		order_depth = state.order_depths[self.symbol]
		buy_orders = sorted(order_depth.buy_orders.items(), reverse = True)
		sell_orders = sorted(order_depth.sell_orders.items())
		
		position = state.position.get(self.symbol, 0)
		buy = self.pos_limit - position
		sell = self.pos_limit + position
		
		self.deque.append(abs(position) == self.pos_limit)
		if len(self.deque) > self.size:
			self.deque.popLeft()

		soft_liquidate = len(self.deque) == self.size and sum(self.deque) >= self.size / 2 and self.deque[-1]
		hard_liquidate = len(self.deque) == self.size and all(self.deque)
		max_buy_price = fair_value - 1 if position > self.pos_limit * 0.5 else fair_value
		min_sell_price = fair_value + 1 if position < self.pos_limit * -0.5 else fair_value

		for price, volume in sell_orders:
			if buy > 0 and price <= max_buy_price:
				quantity = min(buy, -volume)
				self.buy(price, quantity)
				buy -= quantity

			if buy > 0 and hard_liquidate:
				quantity = buy // 2
				self.buy(fair_value, quantity)
				buy -= quantity

			if buy > 0 and soft_liquidate:
				quantity = buy // 2
				self.buy(fair_value - 2, quantity)
				buy -= quantity

			if buy > 0:
				common_buy_price = max(buy_orders, key=lambda x: x[1])[0]
				price = min(max_buy_price, common_buy_price + 1)
				self.buy(price, buy)

		for price, volume in buy_orders:
			if sell > 0 and price >= min_sell_price:
				quantity = min(sell, volume)
				self.sell(price, quantity)
				sell -= quantity

			if sell > 0 and hard_liquidate:
				quantity = sell // 2
				self.sell(fair_value, quantity)
				sell -= quantity

			if sell > 0 and soft_liquidate:
				quantity = sell // 2
				self.sell(fair_value + 2, quantity)
				sell -= quantity

			if sell > 0:
				common_sell_price = min(sell_orders, key=lambda x: x[1])[0]
				price = max(min_sell_price, common_sell_price - 1)
				self.sell(price, sell)
		# Finish implementing
	def save(self) -> JSON:
		return list(self.deque)
	def load(self, load_data: JSON) -> None:
		self.deque = deque(load_data)

class RainforestResinStrategy(TradingStrategy):
	def get_value(self, state: TradingState) -> int:
		return 10000 # Stable, analyze this

class KelpStrategy(TradingStrategy):
	def get_value(self, state: TradingState) -> int:
		order_depth = state.order_depths[self.symbol]
		buy_orders = sorted(order_depth.buy_orders.items(), reverse = True)
		sell_orders = sorted(order_depth.sell_orders.items())
		common_buy_price = max(buy_orders, key = lambda x: x[1])[0]
		common_sell_price = min(sell_orders, key = lambda x: x[1])[0]
		return round((common_buy_price + common_sell_price) / 2) # Not stable, goes up and down over time

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
		conversions = 0
		old_data = json.loads(state.traderData) if state.traderData != "" else {}
		new_data = {}
		orders = {}
		for symbol, strategy in self.strategies.items():
			if symbol in old_data:
				strategy.load(old_data.get(symbol, None))
			if symbol in state.order_depths:
				orders[symbol] = strategy.run(state)
			new_data[symbol] = strategy.save()
		trader_data = json.dumps(new_data, separators=(",", ":"))
		logger.flush(state, orders, conversions, trader_data)
		return orders, conversions, trader_data
    #     print("traderData: " + state.traderData)
    #     print("Observations: " + str(state.observations))

				# # Orders to be placed on exchange matching engine
    #     result = {}
    #     for product in state.order_depths:
    #         order_depth: OrderDepth = state.order_depths[product]
    #         orders: List[Order] = []
    #         acceptable_price = 10  # Participant should calculate this value
    #         print("Acceptable price : " + str(acceptable_price))
    #         print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
    #         if len(order_depth.sell_orders) != 0:
    #             best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
    #             if int(best_ask) < acceptable_price:
    #                 print("BUY", str(-best_ask_amount) + "x", best_ask)
    #                 orders.append(Order(product, best_ask, -best_ask_amount))
    
    #         if len(order_depth.buy_orders) != 0:
    #             best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
    #             if int(best_bid) > acceptable_price:
    #                 print("SELL", str(best_bid_amount) + "x", best_bid)
    #                 orders.append(Order(product, best_bid, -best_bid_amount))
            
    #         result[product] = orders
    
		  #   # String value holding Trader state data required. 
				# # It will be delivered as TradingState.traderData on next execution.
    #     traderData = "SAMPLE" 
        
				# # Sample conversion request. Check more details below. 
    #     conversions = 1
    #     return result, conversions, traderData
