import os
from abc import ABC, abstractmethod
from collections import deque
from datetime import date
from enum import Enum
from itertools import accumulate, pairwise
from operator import mul
from typing import Any, Dict, Iterable, List, Union

import orjson
from loguru import logger
from pydantic import BaseModel, NonNegativeInt

from .memory_db import AccessFeedback, AccessSingle


class TradeAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class SingleAssetPosition(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


def pairwise_diff(input: Iterable) -> List:
    return [b - a for a, b in pairwise(input)]


def element_wise_mul(a: Iterable, b: Iterable) -> List:
    return list(map(mul, a, b))


def cumsum(input: Iterable) -> float:
    return list(accumulate(input))[-1]


class SinglePortfolioDump(BaseModel):
    position: SingleAssetPosition
    symbol: str
    look_back_window_size: int
    trading_dates: List[date]
    trading_price: List[float]
    trading_symbols: List[str]
    trading_position: List[int]
    position_deque: List[int]
    price_deque: List[float]
    evidence_deque: List[List[NonNegativeInt]]


class PortfolioBase(ABC):
    @abstractmethod
    def __init__(self, portfolio_config: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def record_action(
        self,
        action_date: date,
        action: TradeAction,
        price_info: Dict[str, float],
        evidence: List[NonNegativeInt],
    ) -> None:
        pass

    @abstractmethod
    def get_action_record(self) -> List[List[Union[date, float, str]]]:
        pass

    @abstractmethod
    def get_feedback_response(self) -> AccessFeedback:
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load_checkpoint(cls, path: str) -> "PortfolioBase":
        pass


class PortfolioSingleAsset(PortfolioBase):
    def __init__(
        self,
        portfolio_config: Union[Dict[str, Any], None] = None,
        portfolio_dump: Union[SinglePortfolioDump, None] = None,
    ) -> None:
        if portfolio_dump and portfolio_config:
            raise ValueError(
                "Only one of portfolio_config and portfolio_dump should be provided."
            )
        if portfolio_config is not None:
            # init
            self.trading_symbol: str = portfolio_config["trading_symbols"][0]
            logger.trace(f"PORTFOLIO: trading symbol: {self.trading_symbol}")
            self.look_back_window_size = portfolio_config["look_back_window_size"]
            logger.trace(
                f"PORTFOLIO: look back window size: {self.look_back_window_size}"
            )
            # feedback
            self.position_deque = deque(maxlen=self.look_back_window_size)
            self.price_deque = deque(maxlen=self.look_back_window_size + 1)
            self.evidence_deque = deque(maxlen=self.look_back_window_size)
            # records
            self.trading_dates = []
            self.trading_price = []
            self.trading_symbols = []
            self.trading_position = []
            # position
            self.position = SingleAssetPosition.NEUTRAL
            logger.trace(f"PORTFOLIO: initial position: {self.position}")
        elif portfolio_dump is not None:
            self.position = portfolio_dump.position
            self.trading_symbol = portfolio_dump.symbol
            self.look_back_window_size = portfolio_dump.look_back_window_size
            self.position_deque = deque(
                portfolio_dump.position_deque, maxlen=self.look_back_window_size
            )
            self.price_deque = deque(
                portfolio_dump.price_deque, maxlen=self.look_back_window_size + 1
            )
            self.evidence_deque = deque(
                portfolio_dump.evidence_deque, maxlen=self.look_back_window_size
            )
            self.trading_dates = portfolio_dump.trading_dates
            self.trading_price = portfolio_dump.trading_price
            self.trading_symbols = portfolio_dump.trading_symbols
            self.trading_position = portfolio_dump.trading_position
        else:
            raise ValueError(
                "Either portfolio_config or portfolio_dump should be provided."
            )

    def record_action(
        self,
        action_date: date,
        action: TradeAction,
        price_info: Dict[str, float],
        evidence: List[NonNegativeInt],
    ) -> None:
        # transit position
        cur_position = self.position_state_transition(trade_action=action)
        logger.trace(
            f"PORTFOLIO: position transition: {self.position} -> {cur_position} by action: {action}"
        )
        self.position = cur_position
        # append records
        self.trading_dates.append(action_date)
        self.trading_price.append(price_info[self.trading_symbol])
        self.trading_symbols.append(self.trading_symbol)
        self.trading_position.append(self.position.value)
        # register to deque
        self.position_deque.append(cur_position.value)
        logger.trace(f"PORTFOLIO: position deque: {self.position_deque}")
        self.price_deque.append(price_info[self.trading_symbol])
        logger.trace(f"PORTFOLIO: price deque: {self.price_deque}")
        self.evidence_deque.append(evidence)
        logger.trace(f"PORTFOLIO: evidence deque: {self.evidence_deque}")

    def get_feedback_response(self) -> AccessFeedback:
        if len(self.trading_dates) <= self.look_back_window_size:
            return AccessFeedback(access_counter_records=[])
        price_diff = pairwise_diff(input=self.price_deque)
        cumulative_reward = cumsum(
            element_wise_mul(a=price_diff, b=self.position_deque)
        )
        if cumulative_reward == 0:
            return AccessFeedback(access_counter_records=[])
        elif cumulative_reward > 0:
            feedbacks = [AccessSingle(id=i, feedback=1) for i in self.evidence_deque[0]]
            return AccessFeedback(access_counter_records=feedbacks)
        else:
            feedbacks = [
                AccessSingle(id=i, feedback=-1) for i in self.evidence_deque[0]
            ]
            return AccessFeedback(access_counter_records=feedbacks)

    # def get_action_record(self) -> Dict[str, List[Union[date, float, str]]]:
    def get_action_record(
        self,
    ) -> Dict[str, List[date] | List[float] | List[str] | List[int]]:
        return {
            "date": self.trading_dates,
            "price": self.trading_price,
            "symbol": self.trading_symbols,
            "position": self.trading_position,
        }

    @staticmethod
    def position_state_transition(trade_action: TradeAction) -> SingleAssetPosition:
        if trade_action == TradeAction.BUY:
            return SingleAssetPosition.LONG
        elif trade_action == TradeAction.SELL:
            return SingleAssetPosition.SHORT
        else:
            return SingleAssetPosition.NEUTRAL

    def save_checkpoint(self, path: str) -> None:
        # create a dump object
        dump = SinglePortfolioDump(
            position=self.position,
            symbol=self.trading_symbol,
            trading_symbols=self.trading_symbols,
            look_back_window_size=self.look_back_window_size,
            trading_dates=self.trading_dates,
            trading_price=self.trading_price,
            trading_position=self.trading_position,
            position_deque=list(self.position_deque),
            price_deque=list(self.price_deque),
            evidence_deque=list(self.evidence_deque),
        )
        with open(os.path.join(path, "portfolio_checkpoint.json"), "w") as f:
            f.write(orjson.dumps(dump.dict()).decode())

    @classmethod
    def load_checkpoint(cls, path: str) -> "PortfolioSingleAsset":
        with open(os.path.join(path, "portfolio_checkpoint.json"), "r") as f:
            dump = SinglePortfolioDump(**orjson.loads(f.read()))
        return cls(portfolio_dump=dump)

    def __eq__(self, another: "PortfolioSingleAsset") -> bool:
        return all(
            [
                self.position == another.position,
                self.trading_symbol == another.trading_symbol,
                self.look_back_window_size == another.look_back_window_size,
                self.position_deque == another.position_deque,
                self.price_deque == another.price_deque,
                self.evidence_deque == another.evidence_deque,
                self.trading_dates == another.trading_dates,
                self.trading_price == another.trading_price,
                self.trading_symbols == another.trading_symbols,
                self.trading_position == another.trading_position,
            ]
        )


# multi assets
class PortfolioMultiAsset(PortfolioBase):
    def __init__(self, portfolio_config: Dict[str, Any]) -> None:
        pass

    def record_action(
        self,
        action_date: date,
        action: TradeAction,
        price_info: Dict[str, float],
    ) -> None:
        pass

    def get_action_record(self) -> List[List[Union[date, float, str]]]:
        return []

    def get_feedback_response(self) -> AccessFeedback:
        return AccessFeedback(access_counter_records=[])


def construct_portfolio(portfolio_config: Dict[str, Any]) -> PortfolioBase:
    if portfolio_config["type"] != "single-asset":
        raise NotImplementedError
    logger.info("SYS-Portfolio type: single-asset")
    return PortfolioSingleAsset(portfolio_config=portfolio_config)
