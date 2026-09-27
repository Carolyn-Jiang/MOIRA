# sourcery skip: no-loop-in-tests
# sourcery skip: no-conditionals-in-tests

import warnings

warnings.filterwarnings("ignore")

import os
import sys
from typing import Dict

import orjson
import typer
from dotenv import load_dotenv
from loguru import logger
from rich import progress

from src import (
    FinMemAgent,
    MarketEnv,
    RunMode,
    TaskType,
    ensure_path,
    output_metrics_summary,
)

app = typer.Typer()


def load_config(path: str) -> Dict:
    with open(path, "rb") as f:
        return orjson.loads(f.read())


@app.command(name="warmup")
def warmup_up_func(
    config_path: str = typer.Option(
        os.path.join(os.path.dirname(__file__), "configs", "main_fixed_pair.json"), "--config-path", "-c"
    ),
):
    # load config
    config = load_config(path=config_path)

    # ensure path
    ensure_path(save_path=config["meta_config"]["warmup_checkpoint_save_path"])
    ensure_path(save_path=config["meta_config"]["warmup_output_save_path"])
    ensure_path(save_path=config["meta_config"]["log_save_path"])

    # logger
    logger.remove(0)
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "warmup.log"),
        format="{time} {level} {message}",
        level="INFO",
        mode="w",
    )
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "warmup_trace.log"),
        format="{time} {level} {message}",
        level="TRACE",
        mode="w",
    )
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    # log
    logger.info("SYS-Warmup function started")
    logger.info(f"CONFIG-Config path: {config_path}")
    logger.info(f"CONFIG-Config: {config}")

    # init agent
    agent = FinMemAgent(
        agent_config=config["agent_config"],
        emb_config=config["emb_config"],
        chat_config=config["chat_config"],
        portfolio_config=config["portfolio_config"],
    )

    # init env
    env = MarketEnv(
        symbol=config["env_config"]["trading_symbols"],
        env_data_path=config["env_config"]["env_data_path"],
        start_date=config["env_config"]["warmup_start_time"],
        end_date=config["env_config"]["warmup_end_time"],
        momentum_window_size=config["env_config"]["momentum_window_size"],
    )
    # env + agent loop
    total_steps = env.simulation_length
    with progress.Progress() as progress_bar:
        task_id = progress_bar.add_task("Warmup", total=total_steps)
        task = progress_bar.tasks[task_id]
        progress_bar.update(
            task_id, description=f"Warmup remaining: {task.remaining} steps"
        )

        while True:
            logger.info("*" * 50)

            # get obs or terminate
            obs = env.step()
            if obs.termination_flag:
                logger.info("SYS-Environment exhausted.")
                break

            # log
            logger.info("ENV-new info from env")
            logger.info(f"ENV-date: {obs.cur_date}")
            logger.info(f"ENV-price: {obs.cur_price}")
            if obs.cur_news:
                for cur_symbol in obs.cur_news:
                    if obs.cur_news[cur_symbol]:
                        for i, n in enumerate(obs.cur_news[cur_symbol]):  # type: ignore
                            logger.info(f"ENV-news-{cur_symbol}-{i}: {n}")
                            logger.info("-" * 50)
            logger.info(f"ENV-momentum: {obs.cur_momentum}")
            logger.info(f"ENV-symbol: {obs.cur_symbol}")
            logger.info("=" * 50)

            # agent one step
            agent.step(
                market_info=obs, run_mode=RunMode.WARMUP, task_type=TaskType.SingleAsset
            )

            # save checkpoint
            agent.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["warmup_checkpoint_save_path"], "agent"
                )
            )
            env.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["warmup_checkpoint_save_path"], "env"
                )
            )

            # for next iteration
            progress_bar.update(
                task_id,
                advance=1,
                description=f"Warmup remaining steps: {task.remaining}",
            )
    # save warmup results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "agent")
    )
    env.save_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "env")
    )


@app.command(name="warmup-checkpoint")
def warmup_checkpoint_func(
    config_path: str = typer.Option(
        os.path.join(os.path.dirname(__file__), "configs", "main_fixed_pair.json"), "--config-path", "-c"
    ),
):
    # load config
    config = load_config(path=config_path)

    # logger
    logger.remove(0)
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "warmup.log"),
        format="{time} {level} {message}",
        level="INFO",
        mode="a",
    )
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "warmup_trace.log"),
        format="{time} {level} {message}",
        level="TRACE",
        mode="a",
    )
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    # log
    logger.info("SYS-Warmup checkpoint function started")
    logger.info(f"CONFIG-Config path: {config_path}")
    logger.info(f"CONFIG-Config: {config}")

    # load env and agent
    agent = FinMemAgent.load_checkpoint(
        path=os.path.join(
            config["meta_config"]["warmup_checkpoint_save_path"], "agent"
        ),
        task_type=TaskType.SingleAsset,
    )
    env = MarketEnv.load_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_checkpoint_save_path"], "env")
    )

    # env + agent loop
    total_steps = env.simulation_length
    with progress.Progress() as progress_bar:
        task_id = progress_bar.add_task("Warmup", total=total_steps)
        task = progress_bar.tasks[task_id]
        progress_bar.update(
            task_id, description=f"Warmup remaining: {task.remaining} steps"
        )

        while True:
            logger.info("*" * 50)

            # get obs or terminate
            obs = env.step()
            if obs.termination_flag:
                break

            # log
            logger.info("ENV-new info from env")
            logger.info(f"ENV-date: {obs.cur_date}")
            logger.info(f"ENV-price: {obs.cur_price}")
            if obs.cur_news:
                for cur_symbol in obs.cur_news:
                    if obs.cur_news[cur_symbol]:
                        for i, n in enumerate(obs.cur_news[cur_symbol]):  # type: ignore
                            logger.info(f"ENV-news-{cur_symbol}-{i}: {n}")
                            logger.info("-" * 50)
            logger.info(f"ENV-momentum: {obs.cur_momentum}")
            logger.info(f"ENV-symbol: {obs.cur_symbol}")
            logger.info("=" * 50)

            # agent one step
            agent.step(
                market_info=obs, run_mode=RunMode.WARMUP, task_type=TaskType.SingleAsset
            )

            # save checkpoint
            agent.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["warmup_checkpoint_save_path"], "agent"
                )
            )
            env.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["warmup_checkpoint_save_path"], "env"
                )
            )

            # for next iteration
            progress_bar.update(
                task_id,
                advance=1,
                description=f"Warmup remaining steps: {task.remaining}",
            )
    # save warmup results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "agent")
    )
    env.save_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "env")
    )


@app.command(name="test")
def test_func(
    config_path: str = typer.Option(
        os.path.join(os.path.dirname(__file__), "configs", "main_fixed_pair.json"), "--config-path", "-c"
    ),
):
    # load config
    config = load_config(path=config_path)

    # logger
    logger.remove(0)
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "test.log"),
        format="{time} {level} {message}",
        level="INFO",
        mode="w",
    )
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "test_trace.log"),
        format="{time} {level} {message}",
        level="TRACE",
        mode="w",
    )
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    # log
    logger.info("SYS-test function started")
    logger.info(f"CONFIG-Config path: {config_path}")
    logger.info(f"CONFIG-Config: {config}")

    # load env and agent
    agent = FinMemAgent.load_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "agent"),
        task_type=TaskType.SingleAsset,
    )
    env = MarketEnv(
        symbol=config["env_config"]["trading_symbols"],
        env_data_path=config["env_config"]["env_data_path"],
        start_date=config["env_config"]["test_start_time"],
        end_date=config["env_config"]["test_end_time"],
        momentum_window_size=config["env_config"]["momentum_window_size"],
    )

    # env + agent loop
    total_steps = env.simulation_length
    with progress.Progress() as progress_bar:
        task_id = progress_bar.add_task("Warmup", total=total_steps)
        task = progress_bar.tasks[task_id]
        progress_bar.update(
            task_id, description=f"Warmup remaining: {task.remaining} steps"
        )

        while True:
            logger.info("*" * 50)

            # get obs or terminate
            obs = env.step()
            if obs.termination_flag:
                break

            # log
            logger.info("ENV-new info from env")
            logger.info(f"ENV-date: {obs.cur_date}")
            logger.info(f"ENV-price: {obs.cur_price}")
            if obs.cur_news:
                for cur_symbol in obs.cur_news:
                    if obs.cur_news[cur_symbol]:
                        for i, n in enumerate(obs.cur_news[cur_symbol]):  # type: ignore
                            logger.info(f"ENV-news-{cur_symbol}-{i}: {n}")
                            logger.info("-" * 50)
            logger.info(f"ENV-momentum: {obs.cur_momentum}")
            logger.info(f"ENV-symbol: {obs.cur_symbol}")
            logger.info("=" * 50)

            # agent one step
            agent.step(
                market_info=obs, run_mode=RunMode.TEST, task_type=TaskType.SingleAsset
            )

            # save checkpoint
            agent.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["test_checkpoint_save_path"], "agent"
                )
            )
            env.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["test_checkpoint_save_path"], "env"
                )
            )

            # for next iteration
            progress_bar.update(
                task_id,
                advance=1,
                description=f"Warmup remaining steps: {task.remaining}",
            )
    # save results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["test_output_save_path"], "agent")
    )
    env.save_checkpoint(
        path=os.path.join(config["meta_config"]["test_output_save_path"], "env")
    )

    # save final results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["result_save_path"], "agent")
    )


@app.command(name="test-checkpoint")
def test_checkpoint_func(
    config_path: str = typer.Option(
        os.path.join(os.path.dirname(__file__), "configs", "main_fixed_pair.json"), "--config-path", "-c"
    ),
):
    # load config
    config = load_config(path=config_path)

    # logger
    logger.remove(0)
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "test.log"),
        format="{time} {level} {message}",
        level="INFO",
        mode="a",
    )
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "test_trace.log"),
        format="{time} {level} {message}",
        level="TRACE",
        mode="a",
    )
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    # load env and agent
    agent = FinMemAgent.load_checkpoint(
        path=os.path.join(config["meta_config"]["test_checkpoint_save_path"], "agent"),
        task_type=TaskType.SingleAsset,
    )
    env = MarketEnv.load_checkpoint(
        path=os.path.join(config["meta_config"]["test_checkpoint_save_path"], "env"),
    )

    logger.info("SYS-test checkpoint function started")
    logger.info(f"CONFIG-Config path: {config_path}")
    logger.info(f"CONFIG-Config: {config}")

    # env + agent loop
    total_steps = env.simulation_length
    with progress.Progress() as progress_bar:
        task_id = progress_bar.add_task("Warmup", total=total_steps)
        task = progress_bar.tasks[task_id]
        progress_bar.update(
            task_id, description=f"Warmup remaining: {task.remaining} steps"
        )

        while True:
            logger.info("*" * 50)

            # get obs or terminate
            obs = env.step()
            if obs.termination_flag:
                break

            # log
            logger.info("ENV-new info from env")
            logger.info(f"ENV-date: {obs.cur_date}")
            logger.info(f"ENV-price: {obs.cur_price}")
            if obs.cur_news:
                for cur_symbol in obs.cur_news:
                    if obs.cur_news[cur_symbol]:
                        for i, n in enumerate(obs.cur_news[cur_symbol]):  # type: ignore
                            logger.info(f"ENV-news-{cur_symbol}-{i}: {n}")
                            logger.info("-" * 50)
            logger.info(f"ENV-momentum: {obs.cur_momentum}")
            logger.info(f"ENV-symbol: {obs.cur_symbol}")
            logger.info("=" * 50)

            # agent one step
            agent.step(
                market_info=obs, run_mode=RunMode.TEST, task_type=TaskType.SingleAsset
            )

            # save checkpoint
            agent.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["test_checkpoint_save_path"], "agent"
                )
            )
            env.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["test_checkpoint_save_path"], "env"
                )
            )

            # for next iteration
            progress_bar.update(
                task_id,
                advance=1,
                description=f"Warmup remaining steps: {task.remaining}",
            )
    # save results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["test_output_save_path"], "agent")
    )
    env.save_checkpoint(
        path=os.path.join(config["meta_config"]["test_output_save_path"], "env")
    )

    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["result_save_path"], "agent")
    )

@app.command(name="eval")
def eval_func(
    config_path: str = typer.Option(
        os.path.join(os.path.dirname(__file__), "configs", "main_fixed_pair.json"), "--config-path", "-c"
    ),
):
    # load config
    config = load_config(path=config_path)
    output_metrics_summary(
        start_date=config["env_config"]["test_start_time"],
        end_date=config["env_config"]["test_end_time"],
        ticker=config["env_config"]["trading_symbols"][0],
        data_path=list(config["env_config"]["env_data_path"].values())[0],
        result_path=config["meta_config"]["result_save_path"],
        output_path=os.path.join(
            os.path.dirname(config["meta_config"]["result_save_path"]), "metrics"
        ),
    )


if __name__ == "__main__":
    load_dotenv(os.environ.get("FINMEM_ENV_FILE", ".env"), override=True)
    app()
