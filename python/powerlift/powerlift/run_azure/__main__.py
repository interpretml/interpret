""" This is called to run a trial by worker nodes (local / remote). """


def _wait_for_completed_worker(aci_client, resource_group_name, results):
    import time
    import numpy as np

    if len(results) == 0:
        return None

    # if we always start from the beginning we'll eventually fill the first
    # couple of indexes with long running jobs and then have to go through the same
    # ones at the start each time as we look for an opening.  It's better to randomly
    # shuffle the order so we touch all indexes first with equal probability.
    order = np.array(list(results.keys()))[np.random.permutation(len(results))]
    while True:
        for worker_id, result in results.items():
            if result is None:
                del results[worker_id]
                return worker_id
        # for worker_id in order:
        #     result = results[worker_id]
        #     container_group = aci_client.container_groups.get(
        #         resource_group_name, result
        #     )
        #     container = container_group.containers[0]
        #     iview = container.instance_view
        #     if iview is not None:
        #         state = iview.current_state.state
        #         if state == "Terminated":
        #             del results[worker_id]
        #             return worker_id
        time.sleep(1)


def run_azure_process(
    experiment_id,
    n_runners,
    uri,
    timeout,
    raise_exception,
    image,
    azure_json,
    credential,
    num_cores,
    mem_size_gb,
    n_running_containers,
    delete_group_container_on_complete,
    batch_id,
):
    startup_script = """
        is_updated=0
        if ! command -v psql >/dev/null 2>&1; then
            is_updated=1
            apt-get --yes update
            apt-get --yes install postgresql-client
        fi
        shell_install=$(psql "$DB_URL" -c "SELECT shell_install FROM Experiment WHERE id='$EXPERIMENT_ID' LIMIT 1;" -t -A)
        if [ -n "$shell_install" ]; then
            if [ "$is_updated" -eq 0 ]; then
                is_updated=1
                apt-get --yes update
            fi
            cmd="apt-get --yes install $shell_install"
            eval $cmd
        fi
        pip_install=$(psql "$DB_URL" -c "SELECT pip_install FROM Experiment WHERE id='$EXPERIMENT_ID' LIMIT 1;" -t -A)
        if [ -n "$pip_install" ]; then
            cmd="python -m pip install $pip_install"
            eval $cmd
        fi
        filenames=$(psql "$DB_URL" -c "SELECT name FROM wheel WHERE experiment_id='$EXPERIMENT_ID';" -t -A)
        if [ -n "$filenames" ]; then
            echo "$filenames" | while IFS= read -r filename; do
                echo "Processing filename: $filename"
                psql "$DB_URL" -c "COPY (SELECT embedded FROM wheel WHERE experiment_id='$EXPERIMENT_ID' AND name='$filename') TO STDOUT WITH BINARY;" > "$filename"
                cmd="python -m pip install $filename"
                eval $cmd
            done
        fi
        result=$(psql "$DB_URL" -c "SELECT script FROM Experiment WHERE id='$EXPERIMENT_ID' LIMIT 1;" -t -A)
        printf "%s" "$result" > "startup.py"
        python startup.py
    """

    import time
    from azure.mgmt.containerinstance.models import (
        ContainerGroup,
        Container,
        ContainerGroupRestartPolicy,
        EnvironmentVariable,
        ResourceRequests,
        ResourceRequirements,
        OperatingSystemTypes,
    )
    from azure.mgmt.containerinstance import ContainerInstanceManagementClient
    from azure.mgmt.resource import ResourceManagementClient
    from azure.identity import ClientSecretCredential

    if credential is None:
        credential = ClientSecretCredential(
            tenant_id=azure_json["tenant_id"],
            client_id=azure_json["client_id"],
            client_secret=azure_json["client_secret"],
        )

    resource_group_name = azure_json["resource_group"]

    # Run until completion.
    results = {x: None for x in range(n_runners)}
    container_group_names = set()
    resource_group = None
    worker_id = None
    for runner_id in range(n_runners):
        # TODO PK: this entire section has to change because the runners themselves now
        # choose what tasks to work on and this section is no longer responsible
        # for starting new containers

        while True:
            try:
                if resource_group is None:
                    aci_client = ContainerInstanceManagementClient(
                        credential, azure_json["subscription_id"]
                    )
                    res_client = ResourceManagementClient(
                        credential, azure_json["subscription_id"]
                    )
                    resource_group = res_client.resource_groups.get(resource_group_name)

                # worker_id might be non-None if there was an exception we are retrying
                if worker_id is None:
                    worker_id = _wait_for_completed_worker(
                        aci_client, resource_group_name, results
                    )
                    container_group_name = (
                        f"powerlift-container-group-{batch_id}-{worker_id}"
                    )
                else:
                    # if we previously started a container group but had an
                    # error we don't know the state of the container group,
                    # so delete it, if it exists, and restart

                    # TODO: if begin_create_or_update wasn't reached then
                    # there will be no container group to delete so this
                    # will fail, but I don't know yet what exception it
                    # will fail with, so wrap with a try except here
                    # to allow that failure
                    delete_poller = aci_client.container_groups.begin_delete(
                        resource_group_name, container_group_name
                    )
                    while not delete_poller.done():
                        time.sleep(1)

                env_vars = [
                    EnvironmentVariable(name="EXPERIMENT_ID", value=str(experiment_id)),
                    EnvironmentVariable(name="RUNNER_ID", value=str(runner_id)),
                    EnvironmentVariable(name="DB_URL", secure_value=uri),
                    EnvironmentVariable(name="TIMEOUT", value=timeout),
                    EnvironmentVariable(name="RAISE_EXCEPTION", value=raise_exception),
                ]
                container_resource_requests = ResourceRequests(
                    cpu=num_cores,
                    memory_in_gb=mem_size_gb,
                )
                container_resource_requirements = ResourceRequirements(
                    requests=container_resource_requests
                )
                container = Container(
                    name="powerlift-container",
                    image=image,
                    resources=container_resource_requirements,
                    command=["/bin/sh", "-c", startup_script.replace("\r\n", "\n")],
                    environment_variables=env_vars,
                )
                container_group = ContainerGroup(
                    location=resource_group.location,
                    containers=[container],
                    os_type=OperatingSystemTypes.linux,
                    restart_policy=ContainerGroupRestartPolicy.never,
                )

                # begin_create_or_update returns LROPoller,
                # but this is only indicates when the containter is started
                aci_client.container_groups.begin_create_or_update(
                    resource_group.name, container_group_name, container_group
                )

                break
            except:  #  HttpResponseError normally, but I've seen others
                resource_group = None
                time.sleep(1)

        results[worker_id] = container_group_name
        worker_id = None
        container_group_names.add(container_group_name)

    # Wait for all container groups to complete
    while (
        _wait_for_completed_worker(aci_client, resource_group_name, results) is not None
    ):
        pass

    # Delete all container groups
    if delete_group_container_on_complete:
        delete_pollers = []
        for container_group_name in container_group_names:
            delete_poller = aci_client.container_groups.begin_delete(
                resource_group_name, container_group_name
            )
            delete_pollers.append(delete_poller)

        for delete_poller in delete_pollers:
            while not delete_poller.done():
                time.sleep(1)

    return None
