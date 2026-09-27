"""Core type definitions for the redesigned pair-trading framework.

The action space is EXPLICIT and has exactly three members. There is NO ``hold``
action: every trading day the policy must explicitly re-declare its target
position. Maintaining yesterday's position implicitly is forbidden -- if the
policy wants to keep a position it must emit the same explicit action again.

    LONG_SHORT  -> long stock A, short stock B
    SHORT_LONG  -> short stock A, long stock B
    FLAT        -> no exposure (both legs closed)

``PositionState`` mirrors the action space one-to-one because, with no hold
action, the resulting position state is a pure function of the chosen action.
"""

from enum import IntEnum


class Action(IntEnum):
    """Explicit daily target. No HOLD, no CLOSE -- FLAT subsumes closing."""

    long_short = 0  # long leg A, short leg B
    short_long = 1  # short leg A, long leg B
    flat = 2        # no exposure

    @property
    def label(self) -> str:
        return _ACTION_LABELS[self]


class PositionState(IntEnum):
    """Current realized exposure for the active pair (mirror of Action)."""

    long_short = 0
    short_long = 1
    flat = 2

    @property
    def label(self) -> str:
        return _STATE_LABELS[self]


_ACTION_LABELS = {
    Action.long_short: "LONG_SHORT",
    Action.short_long: "SHORT_LONG",
    Action.flat: "FLAT",
}
_STATE_LABELS = {
    PositionState.long_short: "LONG_SHORT",
    PositionState.short_long: "SHORT_LONG",
    PositionState.flat: "FLAT",
}

LABEL_TO_ACTION = {v: k for k, v in _ACTION_LABELS.items()}
# tolerate lower-case / spaced variants from LLM output
LABEL_TO_ACTION.update({k.lower(): v for k, v in LABEL_TO_ACTION.copy().items()})
LABEL_TO_ACTION.update({k.replace("_", " "): v for k, v in list(LABEL_TO_ACTION.items())})

ALL_ACTIONS = (Action.long_short, Action.short_long, Action.flat)


def action_from_label(value) -> Action:
    """Parse an Action from int / Action / string label, rejecting legacy names."""
    if isinstance(value, Action):
        return value
    if isinstance(value, bool):  # guard: bool is a subclass of int
        raise ValueError(f"Invalid action value: {value!r}")
    if isinstance(value, int):
        return Action(value)
    if isinstance(value, str):
        cleaned = value.strip().lower().replace("action.", "").replace(" ", "_")
        # Explicitly reject removed legacy actions so the "No HOLD" rule is enforced.
        if cleaned in ("hold", "close", "long", "short"):
            raise ValueError(
                f"Legacy action '{value}' is not allowed. "
                f"Use LONG_SHORT, SHORT_LONG, or FLAT."
            )
        if cleaned in LABEL_TO_ACTION:
            return LABEL_TO_ACTION[cleaned]
        if cleaned.upper() in LABEL_TO_ACTION:
            return LABEL_TO_ACTION[cleaned.upper()]
    raise ValueError(f"Unknown action: {value!r}")


def action_to_state(action: Action) -> PositionState:
    """With no HOLD action, the post-trade state is determined by the action."""
    return PositionState(int(action))
