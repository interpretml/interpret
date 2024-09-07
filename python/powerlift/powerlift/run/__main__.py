""" This is called to run a trial by worker nodes (local / remote). """


def run_trials(
    experiment_id,
    runner_id,
    db_url,
    timeout,
    raise_exception,
    debug_fn=None,
    is_remote=False,
):
    """Runs trials. Includes wheel installation and timeouts."""
    from powerlift.bench.store import Store
    import traceback
    from powerlift.executors.base import timed_run
    import ast
    import gc

    if is_remote:
        print_exceptions = True
        max_attempts = None
    else:
        print_exceptions = False
        max_attempts = 5

    store = Store(db_url, print_exceptions=print_exceptions, max_attempts=max_attempts)

    if debug_fn is not None:
        trial_run_fn = debug_fn
    else:
        trial_run_fn = store.get_trial_fn(experiment_id)
        trial_run_fn = ast.parse(trial_run_fn)
        if not isinstance(trial_run_fn, ast.Module) or not isinstance(
            trial_run_fn.body[0], ast.FunctionDef
        ):
            raise RuntimeError("Serialized code not valid.")

        func_name = r"wired_function"
        trial_run_fn.body[0].name = func_name
        compiled = compile(trial_run_fn, "<string>", "exec")
        scope = locals()
        exec(compiled, scope, scope)
        trial_run_fn = locals()[func_name]

    while True:
        trial = store.pick_trial(experiment_id, runner_id)
        if trial is None:
            if is_remote:
                print("No more work to start!")
            break

        # Run trial
        errmsg = None
        try:
            # if the previous trial function created cyclic garbage, clear it.
            gc.collect()
            _, duration, timed_out = timed_run(
                lambda: trial_run_fn(trial), timeout_seconds=timeout
            )
            if timed_out:
                raise RuntimeError(f"Timeout failure ({duration})")
        except Exception as e:
            errmsg = f"EXCEPTION: {trial.task.origin}, {trial.task.name}, {trial.method}, {trial.meta}, {trial.task.n_classes}, {trial.task.n_features}, {trial.task.n_samples}\n{traceback.format_exc()}"
            if raise_exception:
                raise e
        finally:
            store.end_trial(trial.id, errmsg)


if __name__ == "__main__":
    print("STARTING POWERLIFT RUNNER")

    import traceback
    import sys

    try:
        import os

        experiment_id = os.getenv("EXPERIMENT_ID")
        runner_id = os.getenv("RUNNER_ID")
        db_url = os.getenv("DB_URL")
        timeout = float(os.getenv("TIMEOUT", 0.0))
        raise_exception = (
            True if os.getenv("RAISE_EXCEPTION", False) == "True" else False
        )
        run_trials(
            experiment_id, runner_id, db_url, timeout, raise_exception, is_remote=True
        )
        sys.exit(0)
    except Exception as e:
        print("EXCEPTION:")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(65)
