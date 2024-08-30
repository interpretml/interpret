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
    from powerlift.bench.store import MIMETYPE_FUNC, BytesParser
    from powerlift.bench.experiment import Store
    import subprocess
    import tempfile
    from pathlib import Path
    import sys

    if is_remote:
        print_exceptions = True
        max_attempts = None
    else:
        print_exceptions = False
        max_attempts = 5

    store = Store(db_url, print_exceptions=print_exceptions, max_attempts=max_attempts)
    while True:
        trial_id = store.pick_trial(experiment_id, runner_id)
        if trial_id is None:
            if is_remote:
                print("No more work to start!")
            break

        trial = store.find_trial_by_id(trial_id)
        if trial is None:
            raise RuntimeError(f"No trial found for id {trial_id}")

        # Handle input assets
        trial_run_fn = None
        for input_asset in trial.input_assets:
            if input_asset.mimetype == MIMETYPE_FUNC:
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
        errmsg = None
        try:
            _, duration, timed_out = timed_run(
                lambda: trial_run_fn(trial), timeout_seconds=timeout
            )
            if timed_out:
                raise RuntimeError(f"Timeout failure ({duration})")
        except Exception as e:
            errmsg = f"EXCEPTION: {trial.task.origin}, {trial.task.name}, {trial.method.name}\n{traceback.format_exc()}"
            if raise_exception:
                raise e
        finally:
            store.end_trial(trial.id, errmsg)


if __name__ == "__main__":
    import os
    import time

    experiment_id = os.getenv("EXPERIMENT_ID")
    runner_id = os.getenv("RUNNER_ID")
    db_url = os.getenv("DB_URL")
    timeout = float(os.getenv("TIMEOUT", 0.0))
    raise_exception = True if os.getenv("RAISE_EXCEPTION", False) == "True" else False
    run_trials(
        experiment_id, runner_id, db_url, timeout, raise_exception, is_remote=True
    )

    # below here is Azure specific. Make optional in the future

    from azure.identity import ManagedIdentityCredential
    from azure.mgmt.containerinstance import ContainerInstanceManagementClient

    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group_name = os.getenv("RESOURCE_GROUP_NAME")
    container_group_name = os.getenv("CONTAINER_GROUP_NAME")

    credential = ManagedIdentityCredential()
    aci_client = ContainerInstanceManagementClient(credential, subscription_id)

    # self-delete the container that we're running on
    delete_poller = aci_client.container_groups.begin_delete(
        resource_group_name, container_group_name
    )
    while not delete_poller.done():
        print("Waiting to be deleted..")
        time.sleep(60)

    print("THIS LINE SHOULD NEVER EXECUTE SINCE THIS CONTAINER SHOULD BE DELETED.")
