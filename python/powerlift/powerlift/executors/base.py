"""Base service classes and methods for services."""

import threading
import time
from numbers import Number
from types import FunctionType
from typing import Any, Optional, Tuple


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

    result = None
    exception = None

    def target():
        nonlocal result, exception
        try:
            result = f()
        except Exception as e:
            exception = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)

    duration = time.time() - start_time
    timed_out = thread.is_alive()

    if exception is not None:
        raise exception

    return result, duration, timed_out


def handle_err(err: Exception):
    """Raises exception provided.

    Args:
        err (Exception): Exception to raise.

    Raises:
        err: Exception provided.
    """
    raise err
