# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

"""Return-value contract for user-supplied training callbacks."""

from enum import Enum


class CallbackAction(Enum):
    """Action a callback signals back to the EBM training loop.

    A callback may return any of the following:

    * ``None`` (i.e. no return statement) — equivalent to ``CONTINUE``.
    * A :class:`CallbackAction` member.
    * The corresponding string value, e.g. ``"stop_current"`` — provided so
      callers do not need to import this module to use the API.
    """

    CONTINUE = "continue"
    """Keep training as usual."""

    STOP_CURRENT = "stop_current"
    """End the current boosting step for this outer bag. Training advances
    to the next big step (e.g. interactions) for this bag normally."""

    STOP_ALL = "stop_all"
    """Stop boosting on all outer bags and end training."""


def _coerce_callback_action(result):
    """Normalize a callback return value into a :class:`CallbackAction`.

    Accepts ``None`` (treated as ``CONTINUE``), a :class:`CallbackAction`
    member, or one of the corresponding string values. Anything else raises
    :class:`TypeError`.
    """
    if result is None:
        return CallbackAction.CONTINUE
    try:
        return CallbackAction(result)
    except ValueError as exc:
        valid = ", ".join(repr(m.value) for m in CallbackAction)
        msg = (
            f"callback returned {result!r}; expected None, a CallbackAction "
            f"member, or one of the string values: {valid}."
        )
        raise TypeError(msg) from exc
