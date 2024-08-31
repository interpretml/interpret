""" This is called to run a trial by worker nodes (local / remote). """


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
        echo "Running startup.py"
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
    from azure.mgmt.authorization import AuthorizationManagementClient
    from azure.mgmt.authorization.models import RoleAssignmentCreateParameters
    from azure.core.exceptions import HttpResponseError
    from multiprocessing.pool import MaybeEncodingError

    import uuid

    client_id = azure_json["client_id"]

    if credential is None:
        credential = ClientSecretCredential(
            tenant_id=azure_json["tenant_id"],
            client_id=client_id,
            client_secret=azure_json["client_secret"],
        )

    resource_group_name = azure_json["resource_group"]
    subscription_id = azure_json["subscription_id"]

    aci_client = ContainerInstanceManagementClient(credential, subscription_id)
    res_client = ResourceManagementClient(credential, subscription_id)
    resource_group = res_client.resource_groups.get(resource_group_name)

    container_resource_requests = ResourceRequests(
        cpu=num_cores,
        memory_in_gb=mem_size_gb,
    )
    container_resource_requirements = ResourceRequirements(
        requests=container_resource_requests
    )

    container_group_names = set()
    starts = []
    for runner_id in range(n_runners):
        container_group_name = f"powerlift-container-group-{batch_id}-{runner_id}"

        env_vars = [
            EnvironmentVariable(name="EXPERIMENT_ID", value=str(experiment_id)),
            EnvironmentVariable(name="RUNNER_ID", value=str(runner_id)),
            EnvironmentVariable(name="DB_URL", secure_value=uri),
            EnvironmentVariable(name="TIMEOUT", value=timeout),
            EnvironmentVariable(name="RAISE_EXCEPTION", value=raise_exception),
            EnvironmentVariable(name="RESOURCE_GROUP_NAME", value=resource_group_name),
            EnvironmentVariable(
                name="CONTAINER_GROUP_NAME", value=container_group_name
            ),
            EnvironmentVariable(name="SUBSCRIPTION_ID", value=subscription_id),
        ]

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
            identity={"type": "SystemAssigned"},
        )

        # begin_create_or_update returns LROPoller,
        # but this is only indicates when the containter is started
        started = aci_client.container_groups.begin_create_or_update(
            resource_group.name, container_group_name, container_group
        )
        starts.append(started)

        container_group_names.add(container_group_name)

    # make sure they have all started before exiting the process
    for started in starts:
        while not started.done():
            time.sleep(1)

    if delete_group_container_on_complete:
        auth_client = AuthorizationManagementClient(credential, subscription_id)

        # Contributor Role
        role_definition_id = f"/subscriptions/{subscription_id}/providers/Microsoft.Authorization/roleDefinitions/b24988ac-6180-42a0-ab88-20f7382dd24c"

        for container_group_name in container_group_names:
            container_group = aci_client.container_groups.get(
                resource_group_name, container_group_name
            )
            scope = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.ContainerInstance/containerGroups/{container_group_name}"
            role_assignment_params = RoleAssignmentCreateParameters(
                role_definition_id=role_definition_id,
                principal_id=container_group.identity.principal_id,
                principal_type="ServicePrincipal",
            )
            auth_client.role_assignments.create(
                scope, str(uuid.uuid4()), role_assignment_params
            )

            role_assignment_params = RoleAssignmentCreateParameters(
                role_definition_id=role_definition_id,
                principal_id=client_id,
                principal_type="User",
            )
            auth_client.role_assignments.create(
                scope, str(uuid.uuid4()), role_assignment_params
            )

        deletes = []
        while len(container_group_names) != 0:
            remove_after = []
            for container_group_name in container_group_names:
                try:
                    container_group = aci_client.container_groups.get(
                        resource_group_name, container_group_name
                    )
                    container = container_group.containers[0]
                    iview = container.instance_view
                    if iview is not None:
                        state = iview.current_state.state
                        if state == "Terminated":
                            # TODO: begin_delete can delete on server but fail
                            # so we should handle the exception that the resource
                            # does not exist
                            deleted = aci_client.container_groups.begin_delete(
                                resource_group_name, container_group_name
                            )
                            deletes.append(deleted)
                            remove_after.append(container_group_name)
                except (HttpResponseError, MaybeEncodingError):
                    pass

            for container_group_name in remove_after:
                container_group_names.remove(container_group_name)

            time.sleep(1)

        for deleted in deletes:
            while not deleted.done():
                time.sleep(1)

    return None
