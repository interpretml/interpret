"""Base service classes and methods for services."""

import time
from numbers import Number
from types import FunctionType
from typing import Any, Optional, Tuple

from stopit import ThreadingTimeout as Timeout


class Executor:
    """Runs the function with time limit.

    Args:
        f: Function to call.
        timeout_seconds: Maximum time limit.
    Returns:
        A tuple containing (response, duration, timed_out as boolean).
    """

    def submit(
        self,
        experiment_id,
        timeout: Optional[int] = None,
    ):
        """Submits and executes trials by trial run function asynchronously.

        Args:
            timeout (int, optional): Timeout in seconds for trial run. Defaults to None.

        Raises:
            NotImplementedError: Raised when executor did not implement this.
        """
        raise NotImplementedError()

    def join(self):
        """Synchronously blocks until execution is complete.

        Raises:
            NotImplementedError: Raised when executor did not implement this.
        """
        raise NotImplementedError()

    def cancel(self):
        """Cancels execution if running.

        Raises:
            NotImplementedError: Raised when executor did not implement this.
        """
        raise NotImplementedError()


def timed_run(f: FunctionType, timeout_seconds: int = 3600) -> Tuple[Any, Number, bool]:
    """Runs the function with time limit.

    Args:
        f (FunctionType): Function to call.
        timeout_seconds (int, optional): Timeout in seconds. Defaults to 3600.

    Returns:
        Tuple[Any, Number, bool]: Function response, duration, has timed out.
    """
    start_time = time.time()
    with Timeout(timeout_seconds) as timeout_ctx:
        res = f()
    duration = time.time() - start_time

    timed_out = timeout_ctx.state == timeout_ctx.TIMED_OUT
    if timed_out:
        res = None
    return res, duration, timed_out


def handle_err(err: Exception):
    """Raises exception provided.

    Args:
        err (Exception): Exception to raise.

    Raises:
        err: Exception provided.
    """
    raise err
