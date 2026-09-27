from datetime import date
from typing import List, Union
from llm.prompt_time import PromptTime, TIME_INSTRUCTION

from ...utils import RunMode
from .base import BasePromptConstructor

# memory layer + id
short_memory_id_desc = "The id of the short-term information."
mid_memory_id_desc = "The id of the mid-term information."
long_memory_id_desc = "The id of the long-term information."
reflection_memory_id_desc = "The id of the reflection-term information."

# Prefix. The fixed pair is represented by one synthetic price index so the
# original FinMem memory and portfolio implementation remain unchanged.
warmup_investment_info_prefix = (
    "The current trading step is {cur_step}. The fixed trading target is {symbol}. "
    "For warm-up reflection only, (current pair index - next-session pair index) "
    "/ current pair index is {future_record}.\n\n"
)
test_investment_info_prefix = (
    "The fixed pair-trading target is {symbol}; the current trading step is {cur_step}. "
    "The pair must remain fixed throughout the experiment: do not select, replace, "
    "or trade any other pair.\n\n"
)

# sentiment  + momentum explanation
sentiment_explanation = """News is tagged by pair leg. Evaluate it comparatively: evidence favorable to the first leg relative to the second supports LONG_SHORT, while evidence favorable to the second leg relative to the first supports SHORT_LONG. Broad or conflicting evidence can support FLAT. Do not infer that positive sentiment for both legs creates a relative-value signal.
                        """
momentum_explanation = """Momentum summarizes the synthetic fixed-pair index over the previous three sessions. Positive momentum means the first-leg-minus-second-leg spread recently increased; negative momentum means it decreased. Treat momentum as one signal rather than a deterministic trading rule.
        """

# summary
warmup_reason = "Given a professional trader's trading suggestion, can you explain to me why the trader drive such a decision with the information provided to you?"
test_reason = "Given the retrieved memories, news, and pair-index momentum, explain the relative-value reason for the fixed-pair decision."

# action
test_action_choice = "Choose one fixed-pair action: LONG_SHORT, SHORT_LONG, or FLAT."

# final prompt
warmup_final_prompt = """Given the following information, can you explain to me why the financial market fluctuation from current day to the next day behaves like this? Summarize the reason of the decision。
        Your should provide a summary information and the id of the information to support your summary."""
test_final_prompt = """Make one decision for the fixed pair and briefly summarize the reason.
    please consider the mid-term information, the long-term information, the reflection-term information only when they are available. If there no such information, directly ignore the impact for absence such information.
    please consider the available short-term information and sentiment associated with them.
    please consider the momentum of the historical pair index and compare evidence across the two legs.
    Output exactly one pair action using this fixed mapping:
    - LONG_SHORT: long the first leg and short the second leg.
    - SHORT_LONG: short the first leg and long the second leg.
    - FLAT: hold no position in either leg.
    Use FLAT when the relative signal is weak, contradictory, or unsupported.
    You also need to provide the ids of the information to support your decision."""


# prompt construction
def _add_momentum_info(momentum: int, investment_info: str) -> str:
    if momentum == -1:
        investment_info += (
            "The cumulative return of the pair index over the past 3 sessions is negative."
        )
    elif momentum == 0:
        investment_info += (
            "The cumulative return of the pair index over the past 3 sessions is zero."
        )
    elif momentum == 1:
        investment_info += (
            "The cumulative return of the pair index over the past 3 sessions is positive."
        )
    return investment_info


class VLLMPromptConstructor(BasePromptConstructor):
    def __init__(self):
        self.prompt_time = PromptTime()

    def __call__(
        self,
        cur_date: date,
        symbol: str,
        run_mode: RunMode,
        future_record: Union[float, None],
        short_memory: Union[List[str], None],
        short_memory_id: Union[List[int], None],
        mid_memory: Union[List[str], None],
        mid_memory_id: Union[List[int], None],
        long_memory: Union[List[str], None],
        long_memory_id: Union[List[int], None],
        reflection_memory: Union[List[str], None],
        reflection_memory_id: Union[List[int], None],
        momentum: Union[int, None] = None,
        cur_step: Union[int, None] = None,
    ) -> str:
        display_symbol = symbol.replace("__", "/")
        step_label = f"step {cur_step:03d}" if cur_step is not None else "step XXX"
        self.prompt_time.labels[cur_date] = step_label
        # investment info + memories
        if run_mode == RunMode.WARMUP:
            investment_info = warmup_investment_info_prefix.format(
                symbol=display_symbol, cur_step=step_label, future_record=future_record
            )
        else:
            investment_info = test_investment_info_prefix.format(
                symbol=display_symbol, cur_step=step_label
            )
        if short_memory and short_memory_id:
            investment_info += "The short-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(short_memory_id, short_memory)]
            )
            investment_info += "\n" + sentiment_explanation
            investment_info += "\n\n"
        if mid_memory and mid_memory_id:
            investment_info += "The mid-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(mid_memory_id, mid_memory)]
            )
            investment_info += "\n\n"
        if long_memory and long_memory_id:
            investment_info += "The long-term information:\n"
            investment_info += "\n".join(
                [f"{i[0]}. {i[1].strip()}" for i in zip(long_memory_id, long_memory)]
            )
            investment_info += "\n\n"
        if reflection_memory and reflection_memory_id:
            investment_info += "The reflection-term information:\n"
            investment_info += "\n".join(
                [
                    f"{i[0]}. {i[1]}"
                    for i in zip(reflection_memory_id, reflection_memory)
                ]
            )
        if momentum:
            investment_info += momentum_explanation
            investment_info = _add_momentum_info(momentum, investment_info)

        suffix = warmup_final_prompt if run_mode == RunMode.WARMUP else test_final_prompt
        grounding = ("Use the supplied warmup outcome only for training reflection."
                     if run_mode == RunMode.WARMUP else TIME_INSTRUCTION)
        return self.prompt_time.text(investment_info + "\n" + suffix + "\n" + grounding)
