""" This is called to run a trial by worker nodes (local / remote). """


def _wait_for_completed_worker(results):
    import time

    if len(results) == 0:
        return None

    while True:
        for worker_id, result in results.items():
            if result is None or result.done():
                del results[worker_id]
                return worker_id
        time.sleep(1)


def run_trials(
    tasks,
    azure_json,
    credential,
    num_cores,
    mem_size_gb,
    n_running_containers,
    delete_group_container_on_complete,
    batch_id,
):
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
    aci_client = ContainerInstanceManagementClient(
        credential, azure_json["subscription_id"]
    )
    res_client = ResourceManagementClient(credential, azure_json["subscription_id"])
    resource_group = res_client.resource_groups.get(resource_group_name)

    # Run until completion.
    container_counter = 0
    n_tasks = len(tasks)
    n_containers = min(n_tasks, n_running_containers)
    results = {x: None for x in range(n_containers)}
    container_group_names = set()
    while len(tasks) != 0:
        params = tasks.pop(0)
        worker_id = _wait_for_completed_worker(results)

        trial_ids, uri, timeout, raise_exception, image = params
        env_vars = [
            EnvironmentVariable(
                name="TRIAL_IDS", value=",".join([str(x) for x in trial_ids])
            ),
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
        container_name = f"powerlift-container-{container_counter}"
        container_counter += 1
        container = Container(
            name=container_name,
            image=image,
            resources=container_resource_requirements,
            command=["python", "-m", "powerlift.run"],
            environment_variables=env_vars,
        )
        container_group = ContainerGroup(
            location=resource_group.location,
            containers=[container],
            os_type=OperatingSystemTypes.linux,
            restart_policy=ContainerGroupRestartPolicy.never,
        )
        container_group_name = f"powerlift-container-group-{worker_id}-{batch_id}"
        result = aci_client.container_groups.begin_create_or_update(
            resource_group.name, container_group_name, container_group
        )

        container_group_names.add(container_group_name)
        results[worker_id] = result

    # Wait for all container groups to complete
    while _wait_for_completed_worker(results) is not None:
        pass

    # Delete all container groups
    if delete_group_container_on_complete:
        for container_group_name in container_group_names:
            aci_client.container_groups.begin_delete(
                resource_group_name, container_group_name
            )
    return None
