"""Engle--Granger-guided Double DQN baseline for discrete pair trading.

The statistical layer fixes a causally selected pair.  DDQN learns only the daily
direction decision from numeric price/spread features; it receives no news, language
model output, policy memory, lifecycle state, or test-period updates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.baselines import LONG_SHORT, SHORT_LONG, settle_baseline
from utils.engine import _zeroed_targets
from utils.metrics import compute_performance_metrics
from utils.portfolio import (
    Action,
    PortfolioManager,
    effective_hedge_ratio,
    size_pair_position_fractional,
)


ACTIONS = (LONG_SHORT, "FLAT", SHORT_LONG)


def _market_features(history: Sequence[Tuple[float, float]], lookback: int = 20) -> Tuple[np.ndarray, float]:
    """Return eight causal numeric features and the current OLS hedge ratio."""
    values = np.asarray(history[-lookback:], dtype=float)
    if len(values) < 3:
        return np.zeros(8, dtype=float), 1.0
    la, lb = np.log(values[:, 0]), np.log(values[:, 1])
    var_b = float(np.var(lb))
    beta = float(np.cov(la, lb, ddof=0)[0, 1] / var_b) if var_b > 1e-12 else 1.0
    beta = float(np.clip(beta, -5.0, 5.0))
    spread = la - beta * lb
    sd = float(np.std(spread))
    z = float((spread[-1] - np.mean(spread)) / sd) if sd > 1e-12 else 0.0
    prev_z = float((spread[-2] - np.mean(spread[:-1])) / np.std(spread[:-1])) \
        if len(spread) > 3 and np.std(spread[:-1]) > 1e-12 else z
    ra = float(la[-1] - la[-2])
    rb = float(lb[-1] - lb[-2])
    momentum = float((la[-1] - la[max(0, len(la) - 6)]) -
                     (lb[-1] - lb[max(0, len(lb) - 6)]))
    corr = float(np.corrcoef(np.diff(la), np.diff(lb))[0, 1]) if len(la) > 3 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    return np.asarray([z, z - prev_z, beta, sd, ra, rb, momentum, corr]), beta


def build_market_series(
    dates: Sequence,
    env_data: Dict,
    pair: Tuple[str, str],
    *,
    history_seed: Optional[Dict] = None,
    lookback: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    history: List[Tuple[float, float]] = []
    if history_seed:
        for day in sorted(history_seed):
            prices = history_seed[day]["price"]
            if pair[0] in prices and pair[1] in prices:
                history.append((float(prices[pair[0]]), float(prices[pair[1]])))
    features, betas = [], []
    for day in dates:
        prices = env_data[day]["price"]
        history.append((float(prices[pair[0]]), float(prices[pair[1]])))
        feat, beta = _market_features(history, lookback)
        features.append(feat)
        betas.append(beta if len(history) >= max(5, lookback // 2) else 1.0)
    return np.asarray(features), np.asarray(betas)


def _action_return(prices_t, prices_next, pair, action_index, beta) -> float:
    if ACTIONS[action_index] == "FLAT":
        return 0.0
    pa, pb = float(prices_t[pair[0]]), float(prices_t[pair[1]])
    b = effective_hedge_ratio(beta)
    sign = 1.0 if ACTIONS[action_index] == LONG_SHORT else -1.0
    # Match the shared engine's share ratio q_B / q_A = beta and gross exposure.
    return sign * ((float(prices_next[pair[0]]) - pa)
                   - b * (float(prices_next[pair[1]]) - pb)) / (pa + b * pb)



class _Network:
    def __init__(self, input_dim: int, hidden: int, output_dim: int, rng: np.random.Generator):
        self.params = {
            "w1": rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0, np.sqrt(2 / hidden), (hidden, output_dim)),
            "b2": np.zeros(output_dim),
        }
        self.m = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.v = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.steps = 0

    def copy_from(self, other: "_Network") -> None:
        self.params = {key: value.copy() for key, value in other.params.items()}

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, x @ self.params["w1"] + self.params["b1"])
        return h @ self.params["w2"] + self.params["b2"]

    def train(self, x: np.ndarray, actions: np.ndarray, targets: np.ndarray, lr: float) -> float:
        z1 = x @ self.params["w1"] + self.params["b1"]
        h = np.maximum(0.0, z1)
        q = h @ self.params["w2"] + self.params["b2"]
        errors = q[np.arange(len(x)), actions] - targets
        clipped = np.clip(errors, -1.0, 1.0) / len(x)
        dq = np.zeros_like(q)
        dq[np.arange(len(x)), actions] = clipped
        grads = {
            "w2": h.T @ dq,
            "b2": np.sum(dq, axis=0),
        }
        dh = dq @ self.params["w2"].T
        dz = dh * (z1 > 0)
        grads["w1"] = x.T @ dz
        grads["b1"] = np.sum(dz, axis=0)
        self.steps += 1
        for key, grad in grads.items():
            grad = np.clip(grad, -5.0, 5.0)
            self.m[key] = 0.9 * self.m[key] + 0.1 * grad
            self.v[key] = 0.999 * self.v[key] + 0.001 * grad * grad
            mh = self.m[key] / (1.0 - 0.9 ** self.steps)
            vh = self.v[key] / (1.0 - 0.999 ** self.steps)
            self.params[key] -= lr * mh / (np.sqrt(vh) + 1e-8)
        return float(np.mean(errors * errors))


@dataclass
class DDQNConfig:
    episodes: int = 600
    hidden: int = 32
    batch_size: int = 64
    replay_capacity: int = 5000
    gamma: float = 0.95
    learning_rate: float = 3e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    target_update: int = 100
    reward_scale: float = 100.0


class EGDoubleDQN:
    def __init__(self, seed: int, config: DDQNConfig = DDQNConfig()):
        self.seed, self.config = int(seed), config
        self.rng = np.random.default_rng(seed)
        self.online = _Network(11, config.hidden, 3, self.rng)
        self.target = _Network(11, config.hidden, 3, self.rng)
        self.target.copy_from(self.online)
        self.mean = np.zeros(8)
        self.std = np.ones(8)

    def _state(self, market: np.ndarray, position: int) -> np.ndarray:
        one_hot = np.zeros(3)
        one_hot[position] = 1.0
        return np.concatenate(((market - self.mean) / self.std, one_hot))

    def fit(self, dates, env_data, pair, *, beta_hedged: bool, history_seed=None) -> Dict[str, float]:
        if len(dates) < 2 or self.config.episodes < 1:
            raise ValueError("DDQN needs at least two training observations and one episode")
        market, betas = build_market_series(dates, env_data, pair, history_seed=history_seed)
        self.mean = market.mean(axis=0)
        self.std = market.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        replay = []
        updates, final_loss = 0, 0.0
        total_steps = max(1, self.config.episodes * (len(dates) - 1))
        for episode in range(self.config.episodes):
            position = 1
            for t in range(len(dates) - 1):
                progress = (episode * (len(dates) - 1) + t) / total_steps
                epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * (1 - progress)
                state = self._state(market[t], position)
                action = int(self.rng.integers(3)) if self.rng.random() < epsilon else int(np.argmax(self.online.predict(state[None])[0]))
                beta = betas[t] if beta_hedged else 1.0
                reward = _action_return(env_data[dates[t]]["price"], env_data[dates[t + 1]]["price"], pair, action, beta)
                next_state = self._state(market[t + 1], action)
                replay.append((state, action, reward * self.config.reward_scale, next_state, t == len(dates) - 2))
                if len(replay) > self.config.replay_capacity:
                    replay.pop(0)
                position = action
                if len(replay) >= self.config.batch_size:
                    idx = self.rng.integers(0, len(replay), self.config.batch_size)
                    batch = [replay[i] for i in idx]
                    states = np.asarray([row[0] for row in batch])
                    actions = np.asarray([row[1] for row in batch])
                    rewards = np.asarray([row[2] for row in batch])
                    next_states = np.asarray([row[3] for row in batch])
                    dones = np.asarray([row[4] for row in batch], dtype=float)
                    next_actions = np.argmax(self.online.predict(next_states), axis=1)
                    next_q = self.target.predict(next_states)[np.arange(len(batch)), next_actions]
                    targets = rewards + self.config.gamma * (1.0 - dones) * next_q
                    final_loss = self.online.train(states, actions, targets, self.config.learning_rate)
                    updates += 1
                    if updates % self.config.target_update == 0:
                        self.target.copy_from(self.online)
        return {"updates": updates, "final_loss": final_loss}

    def evaluate(self, dates, env_data, symbols, pair, *, beta_hedged: bool, history_seed=None, settlement_date=None):
        market, betas = build_market_series(dates, env_data, pair, history_seed=history_seed)
        portfolio = PortfolioManager(list(symbols), initial_cash=1.0)
        equity, actions = [], []
        position = 1
        for t, day in enumerate(dates):
            prices = env_data[day]["price"]
            state = self._state(market[t], position)
            position = int(np.argmax(self.online.predict(state[None])[0]))
            action = ACTIONS[position]
            targets = _zeroed_targets(symbols)
            if action != "FLAT":
                pv = max(portfolio._compute_equity(prices), 1e-9)
                beta = effective_hedge_ratio(betas[t], pair=pair, as_of=day.isoformat()) if beta_hedged else 1.0
                targets.update(size_pair_position_fractional(
                    prices, pair, pv,
                    Action.long_short if action == LONG_SHORT else Action.short_long,
                    hedge_ratio=beta,
                ))
            portfolio.step(prices, targets)
            equity.append(float(portfolio.normalized_equity_history[-1]))
            actions.append(action)
        if settlement_date is not None:
            equity.append(settle_baseline(portfolio, dates, env_data, symbols, settlement_date))
        metrics = compute_performance_metrics(pd.Series(equity, dtype=float), 1.0)
        return {
            "normalized_equity": equity,
            "actions": actions,
            "cum_return": equity[-1] - 1.0,
            "sharpe": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
        }
