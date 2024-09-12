""" This is called to run a trial by worker nodes (local / remote). """


def run_trials(
    experiment_id,
    runner_id,
    db_url,
    timeout,
    raise_exception,
    print_exceptions=False,
    max_attempts=5,
    return_after_one=False,
):
    """Runs trials. Includes wheel installation and timeouts."""
    from powerlift.bench.store import Store
    import traceback
    from powerlift.executors.base import timed_run
    import ast
    import gc

    store = Store(db_url, print_exceptions=print_exceptions, max_attempts=max_attempts)

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
        errmsg = "UNKNOWN FAILURE"

        trial = store.pick_trial(experiment_id, runner_id)
        if trial is None:
            if print_exceptions:
                print("No more work to start!")
            return True

        # Run trial
        try:
            # if the previous trial function created cyclic garbage, clear it.
            gc.collect()
            _, duration, timed_out = timed_run(
                lambda: trial_run_fn(trial), timeout_seconds=timeout
            )
            if timed_out:
                raise RuntimeError(f"Timeout failure ({duration})")
            errmsg = None
        except Exception:
            errmsg = f"EXCEPTION: {trial.task.origin}, {trial.task.name}, {trial.method}, {trial.meta}, {trial.task.n_classes}, {trial.task.n_features}, {trial.task.n_samples}\n{traceback.format_exc()}"
            if raise_exception:
                raise
        except BaseException:
            errmsg = f"EXCEPTION: {trial.task.origin}, {trial.task.name}, {trial.method}, {trial.meta}, {trial.task.n_classes}, {trial.task.n_features}, {trial.task.n_samples}\n{traceback.format_exc()}"
            raise
        finally:
            store.end_trial(trial.id, errmsg)

        if return_after_one:
            return False


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
        is_done = run_trials(
            experiment_id,
            runner_id,
            db_url,
            timeout,
            False,
            True,
            None,
            True,
        )
    except Exception as e:
        print("EXCEPTION:")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(65)
    except BaseException as e:
        print("EXCEPTION:")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        sys.exit(64)

    sys.exit(0 if is_done else 1)
