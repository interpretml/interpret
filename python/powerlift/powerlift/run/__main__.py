""" This is called to run a trial by worker nodes (local / remote). """


def run_trials(trial_ids, db_url, timeout, raise_exception, debug_fn=None):
    """Runs trials. Includes wheel installation and timeouts."""
    from powerlift.bench.store import Store
    import traceback
    from powerlift.executors.base import timed_run
    from powerlift.bench.store import MIMETYPE_FUNC, MIMETYPE_WHEEL, BytesParser
    from powerlift.bench.experiment import Store
    import subprocess
    import tempfile
    from pathlib import Path
    import sys

    store = Store(db_url)
    for trial_id in trial_ids:
        trial = store.find_trial_by_id(trial_id)
        if trial is None:
            raise RuntimeError(f"No trial found for id {trial_id}")

        # Handle input assets
        trial_run_fn = None
        with tempfile.TemporaryDirectory() as dirpath:
            for input_asset in trial.input_assets:
                if input_asset.mimetype == MIMETYPE_WHEEL:
                    wheel = BytesParser.deserialize(
                        input_asset.mimetype, input_asset.embedded
                    )
                    filepath = Path(dirpath, input_asset.name)
                    with open(filepath, "wb") as f:
                        f.write(wheel.content)
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", filepath]
                    )
                elif input_asset.mimetype == MIMETYPE_FUNC:
                    trial_run_fn = BytesParser.deserialize(
                        MIMETYPE_FUNC, input_asset.embedded
                    )
                else:
                    continue
        if debug_fn is not None:
            trial_run_fn = debug_fn

        if trial_run_fn is None:
            raise RuntimeError("No trial run function found.")

        # Run trial
        store.start_trial(trial.id)
        errmsg = None
        try:
            _, duration, timed_out = timed_run(
                lambda: trial_run_fn(trial), timeout_seconds=timeout
            )
            if timed_out:
                raise RuntimeError(f"Timeout failure ({duration})")
        except Exception as e:
            errmsg = traceback.format_exc()
            if raise_exception:
                raise e
        finally:
            store.end_trial(trial.id, errmsg)


if __name__ == "__main__":
    import os

    trial_ids = os.getenv("TRIAL_IDS").split(",")
    db_url = os.getenv("DB_URL")
    timeout = float(os.getenv("TIMEOUT", 0.0))
    raise_exception = True if os.getenv("RAISE_EXCEPTION", False) == "True" else False

    run_trials(trial_ids, db_url, timeout, raise_exception)
