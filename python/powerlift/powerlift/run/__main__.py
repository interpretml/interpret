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
        trial_id = store.pick_trial(experiment_id, runner_id)
        if trial_id is None:
            if is_remote:
                print("No more work to start!")
            break

        trial = store.find_trial_by_id(trial_id)
        if trial is None:
            raise RuntimeError(f"No trial found for id {trial_id}")

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
    print("STARTING RUNNER")

    import time
    import traceback

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
    except Exception as e:
        print("EXCEPTION:")
        print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        for _ in range(60 * 60 * 24):  # wait 24 hours
            time.sleep(1)
            print("Unandled exception.")
