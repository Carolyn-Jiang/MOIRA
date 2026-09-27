from datetime import date
from pathlib import Path
import sys

import pandas as pd
import pytest

BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))
pytest.importorskip("loguru", reason="FinMem optional dependencies are not installed")

from scripts.build_fixed_pair_data import build_records
from src.chat.endpoint.base import StructureOutputResponse
from src.chat.prompt.vllm_prompt import VLLMPromptConstructor
from src.eval_pipeline import evaluate_fixed_pair
from src.portfolio import TradeAction
from src.utils import RunMode


def test_pair_output_labels_preserve_finmem_enum():
    long_short = StructureOutputResponse(
        investment_decision="LONG_SHORT", summary_reason="relative strength"
    )
    short_long = StructureOutputResponse(
        investment_decision="SHORT_LONG", summary_reason="relative weakness"
    )
    flat = StructureOutputResponse(
        investment_decision="FLAT", summary_reason="weak signal"
    )
    assert (long_short.investment_decision, long_short.pair_action) == (
        TradeAction.BUY,
        "LONG_SHORT",
    )
    assert (short_long.investment_decision, short_long.pair_action) == (
        TradeAction.SELL,
        "SHORT_LONG",
    )
    assert (flat.investment_decision, flat.pair_action) == (TradeAction.HOLD, "FLAT")


def test_prompt_fixes_pair_and_defines_all_actions():
    prompt = VLLMPromptConstructor()(
        cur_date=date(2025, 2, 12),
        symbol="BMRN__META",
        run_mode=RunMode.TEST,
        future_record=None,
        short_memory=None,
        short_memory_id=None,
        mid_memory=None,
        mid_memory_id=None,
        long_memory=None,
        long_memory_id=None,
        reflection_memory=None,
        reflection_memory_id=None,
        momentum=1,
    )
    assert "BMRN/META" in prompt
    assert "do not select, replace" in prompt
    assert "LONG_SHORT: long the first leg" in prompt
    assert "SHORT_LONG: short the first leg" in prompt
    assert "Use FLAT when" in prompt


def test_builder_and_evaluator_use_two_leg_returns():
    prices_a = {"2025-01-01": 100.0, "2025-01-02": 110.0, "2025-01-03": 110.0}
    prices_b = {"2025-01-01": 100.0, "2025-01-02": 100.0, "2025-01-03": 110.0}
    records = build_records(
        prices_a,
        prices_b,
        {},
        {},
        "BMRN",
        "META",
        "2025-01-01",
        "2025-01-03",
    )
    actions = pd.DataFrame(
        {
            "date": [date(2025, 1, 1), date(2025, 1, 2)],
            "direction": [1, -1],
        }
    )
    daily, metrics = evaluate_fixed_pair(
        records, actions, "2025-01-01", "2025-01-03"
    )
    no_beta = daily[daily["sizing"] == "No-Beta"]
    assert no_beta["pair_action"].tolist() == ["LONG_SHORT", "SHORT_LONG"]
    assert no_beta["daily_return"].tolist() == [0.05, 10.0 / 210.0]
    assert set(metrics["sizing"]) == {"No-Beta", "Beta-Hedged"}


def test_raw_news_builder_excludes_same_day_news():
    prices = {"2025-02-14": 100., "2025-02-18": 101., "2025-02-19": 102.}
    news = {"2025-02-14": {"news": ["Friday"]},
            "2025-02-17": {"news": ["holiday"]},
            "2025-02-18": {"news": ["not available yet"]}}
    records = build_records(prices, prices, news, {}, "A", "B", "2025-02-14", "2025-02-19")
    assert records["2025-02-18"]["news"] == ["[A] Friday", "[A] holiday"]


def test_missing_decisions_do_not_silently_become_flat():
    prices = {"2025-02-12": 100., "2025-02-13": 110., "2025-02-14": 105.}
    records = build_records(prices, prices, {}, {}, "A", "B", "2025-02-12", "2025-02-14")
    actions = pd.DataFrame({"date": ["2025-02-12"], "direction": [1]})
    with pytest.raises(ValueError, match="Missing FinMem decisions"):
        evaluate_fixed_pair(records, actions, "2025-02-12", "2025-02-14")


def test_market_env_75_decisions_and_price_only_terminal(tmp_path):
    import json
    from src.market_env import MarketEnv
    dates = [d.date().isoformat() for d in pd.bdate_range("2025-02-12", "2025-05-30")
             if d.date().isoformat() not in {"2025-02-17", "2025-04-18", "2025-05-26"}]
    records = {d: {"prices": 100., "news": []} for d in [*dates, "2025-06-02"]}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(records))
    env = MarketEnv(env_data_path={"A__B": str(path)}, start_date=dates[0],
                    end_date="2025-06-02", symbol=["A__B"], momentum_window_size=3)
    decisions = []
    while True:
        obs = env.step()
        if obs.termination_flag:
            break
        decisions.append(obs.cur_date.isoformat())
    assert len(decisions) == 75
    assert decisions[-1] == "2025-05-30"


def test_memory_checkpoint_restores_with_mocked_inference(tmp_path, monkeypatch):
    import json
    from src.agent import FinMemAgent
    from src.embedding import OpenAIEmbedding
    from src.chat.endpoint.openai import OpenAIStructureGeneration
    from src.market_env import OneDayMarketInfo
    from src.utils import TaskType
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-placeholder")
    monkeypatch.setattr(OpenAIEmbedding, "__call__",
                        lambda self, texts: [[1., 0., 0., 0.] for _ in ([texts] if isinstance(texts, str) else texts)])
    monkeypatch.setattr(OpenAIStructureGeneration, "__call__",
                        lambda self, prompt, schema: StructureOutputResponse(
                            investment_decision="LONG_SHORT", summary_reason="Synthetic test evidence"))
    config = json.loads((BASELINE_ROOT / "configs/main_fixed_pair.json").read_text())
    config["emb_config"]["emb_size"] = 4
    config["agent_config"]["memory_db_config"]["memory_db_endpoint"] = f"local:{tmp_path / 'db'}"
    agent = FinMemAgent(**{key: config[key] for key in (
        "agent_config", "emb_config", "chat_config", "portfolio_config")})
    symbol = config["env_config"]["trading_symbols"][0]
    obs = OneDayMarketInfo(cur_date=date(2025, 2, 12), cur_price={symbol: 100.},
        cur_news={symbol: ["Synthetic news"]}, cur_future_price_diff={symbol: .01},
        cur_momentum={symbol: None}, cur_symbol=[symbol], termination_flag=False)
    try:
        agent.step(obs, RunMode.WARMUP, TaskType.SingleAsset)
        agent.save_checkpoint(str(tmp_path / "checkpoint"))
    finally:
        agent.memory_db.connection_client.close()
    restored = FinMemAgent.load_checkpoint(str(tmp_path / "checkpoint"), TaskType.SingleAsset)
    try:
        obs.cur_date = date(2025, 2, 13)
        restored.step(obs, RunMode.TEST, TaskType.SingleAsset)
        restored.save_checkpoint(str(tmp_path / "restored"))
        assert (tmp_path / "restored/portfolio_checkpoint.json").exists()
    finally:
        restored.memory_db.connection_client.close()
