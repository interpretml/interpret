"""This is called to run a trial by worker nodes (local / remote)."""


def assign_contributor_permissions(
    aci_client,
    auth_client,
    max_undead,
    credential,
    subscription_id,
    client_id,
    resource_group_name,
    resource_uris,
    container_groups,
):
    from heapq import heappush, heappop
    from datetime import datetime
    import time
    import uuid
    from azure.mgmt.containerinstance import ContainerInstanceManagementClient
    from azure.mgmt.authorization import AuthorizationManagementClient
    from azure.mgmt.authorization.models import RoleAssignmentCreateParameters
    from azure.core.exceptions import HttpResponseError

    # Contributor Role
    contributor_definition_id = f"/subscriptions/{subscription_id}/providers/Microsoft.Authorization/roleDefinitions/b24988ac-6180-42a0-ab88-20f7382dd24c"

    while max_undead < len(container_groups):
        _, container_group_name, started = heappop(container_groups)
        try:
            if started is not None:
                if not started.done():
                    heappush(
                        container_groups,
                        (datetime.now(), container_group_name, started),
                    )
                    time.sleep(1)
                    continue
                started = None

            if aci_client is None:
                aci_client = ContainerInstanceManagementClient(
                    credential, subscription_id
                )

            container_group = aci_client.container_groups.get(
                resource_group_name, container_group_name
            )

            ra_principal_contributor = RoleAssignmentCreateParameters(
                role_definition_id=contributor_definition_id,
                principal_id=container_group.identity.principal_id,
                principal_type="ServicePrincipal",
            )
            ra_user_contributor = RoleAssignmentCreateParameters(
                role_definition_id=contributor_definition_id,
                principal_id=client_id,
                principal_type="User",
            )

            scope = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.ContainerInstance/containerGroups/{container_group_name}"

            if auth_client is None:
                auth_client = AuthorizationManagementClient(credential, subscription_id)

            auth_client.role_assignments.create(
                scope, str(uuid.uuid4()), ra_principal_contributor
            )
            if resource_uris is not None:
                for resource_uri in resource_uris:
                    auth_client.role_assignments.create(
                        resource_uri, str(uuid.uuid4()), ra_principal_contributor
                    )

            auth_client.role_assignments.create(
                scope, str(uuid.uuid4()), ra_user_contributor
            )
        except HttpResponseError:
            aci_client = None
            auth_client = None
            heappush(container_groups, (datetime.now(), container_group_name, started))
            time.sleep(1)

    return aci_client, auth_client


def run_azure_process(
    experiment_id,
    n_instances,
    uri,
    resource_uris,
    timeout,
    image,
    azure_json,
    credential,
    num_cores,
    mem_size_gb,
    max_undead,
    delete_on_complete,
    batch_id,
):
    startup_script = """
        self_delete() {
            echo "Attempt to self-delete this container group. Exit code was: $1"

            if [ $1 -ne 0 ]; then
                echo "Waiting 10 mintues to allow inspection of the logs..."
                sleep 600
            fi

            SUBSCRIPTION_ID=${SUBSCRIPTION_ID}
            RESOURCE_GROUP_NAME=${RESOURCE_GROUP_NAME}
            CONTAINER_GROUP_NAME=${CONTAINER_GROUP_NAME}

            retry_count=0
            while true; do
                echo "Downloading azure tools."

                curl -sL https://aka.ms/InstallAzureCLIDeb -o install_script.sh
                exit_code=$?
                if [ $exit_code -ne 0 ]; then
                    echo "curl failed with exit code $exit_code."
                fi

                bash install_script.sh
                exit_code=$?
                if [ $exit_code -ne 0 ]; then
                    echo "Failed to install azure tools with exit code $exit_code. Attempting to delete anyway."
                fi

                echo "Logging into azure to delete this container."
                az login --identity
                exit_code=$?
                if [ $exit_code -ne 0 ]; then
                    echo "az login failed with exit code $exit_code."
                fi

                echo "Deleting the container."
                az container delete --subscription $SUBSCRIPTION_ID --resource-group $RESOURCE_GROUP_NAME --name $CONTAINER_GROUP_NAME --yes
                exit_code=$?
                if [ $exit_code -ne 0 ]; then
                    echo "az container delete failed with exit code $exit_code."
                fi

                echo "Waiting to be deleted..."
                sleep 300

                if [ $retry_count -ge 300 ]; then
                    echo "Maximum number of retries reached. Cannot self-delete this container group."
                    break
                fi
                retry_count=$((retry_count + 1))

                echo "Retrying."
            done
            
            exit $1  # failed to self-kill the container we are running this on.
        }
    
        is_updated=0
        if ! command -v psql >/dev/null 2>&1; then
            is_updated=1
            apt-get --yes update
            exit_code=$?
            if [ $exit_code -ne 0 ]; then
                echo "apt-get --yes update failed with exit code $exit_code."
                self_delete $exit_code
            fi
            apt-get --yes install postgresql-client
            exit_code=$?
            if [ $exit_code -ne 0 ]; then
                echo "apt-get --yes install postgresql-client failed with exit code $exit_code."
                self_delete $exit_code
            fi
        fi
        retry_count=0
        while true; do
            shell_install=$(psql "$DB_URL" -c "SELECT shell_install FROM Experiment WHERE id = '$EXPERIMENT_ID' LIMIT 1;" -t -A)
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                break
            fi
            echo "psql failed with exit code $exit_code."
            if [ $retry_count -ge 300 ]; then
                echo "Maximum number of retries reached. Command failed."
                self_delete $exit_code
            fi
            retry_count=$((retry_count + 1))
            echo "Sleeping."
            sleep 300
            echo "Retrying."
        done
        if [ -n "$shell_install" ]; then
            if [ $is_updated -eq 0 ]; then
                is_updated=1
                apt-get --yes update
                exit_code=$?
                if [ $exit_code -ne 0 ]; then
                    echo "apt-get --yes update failed with exit code $exit_code."
                    self_delete $exit_code
                fi
            fi
            cmd="apt-get --yes install $shell_install"
            eval $cmd
            exit_code=$?
            if [ $exit_code -ne 0 ]; then
                echo "apt-get --yes install shell_install failed with exit code $exit_code."
                self_delete $exit_code
            fi
        fi

        retry_count=0
        while true; do
            filenames=$(psql "$DB_URL" -c "SELECT name FROM wheel WHERE experiment_id = '$EXPERIMENT_ID';" -t -A)
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                break
            fi
            echo "psql failed with exit code $exit_code."
            if [ $retry_count -ge 300 ]; then
                echo "Maximum number of retries reached. Command failed."
                self_delete $exit_code
            fi
            retry_count=$((retry_count + 1))
            echo "Sleeping."
            sleep 300
            echo "Retrying."
        done
        if [ -n "$filenames" ]; then
            echo "$filenames" | while IFS= read -r filename; do
                echo "Processing filename: $filename"
                retry_count=0
                while true; do
                    psql "$DB_URL" -c "COPY (SELECT embedded FROM wheel WHERE experiment_id = '$EXPERIMENT_ID' AND name = '$filename') TO STDOUT WITH BINARY;" > "$filename"
                    exit_code=$?
                    if [ $exit_code -eq 0 ]; then
                        break
                    fi
                    echo "psql failed with exit code $exit_code."
                    if [ $retry_count -ge 300 ]; then
                        echo "Maximum number of retries reached. Command failed."
                        self_delete $exit_code
                    fi
                    retry_count=$((retry_count + 1))
                    echo "Sleeping."
                    sleep 300
                    echo "Retrying."
                done
                cmd="python -m pip install $filename"
                eval $cmd
                exit_code=$?
                if [ $exit_code -ne 0 ]; then
                    echo "python -m pip install filename failed with exit code $exit_code."
                    self_delete $exit_code
                fi
            done
        fi

        retry_count=0
        while true; do
            pip_install=$(psql "$DB_URL" -c "SELECT pip_install FROM Experiment WHERE id = '$EXPERIMENT_ID' LIMIT 1;" -t -A)
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                break
            fi
            echo "psql failed with exit code $exit_code."
            if [ $retry_count -ge 300 ]; then
                echo "Maximum number of retries reached. Command failed."
                self_delete $exit_code
            fi
            retry_count=$((retry_count + 1))
            echo "Sleeping."
            sleep 300
            echo "Retrying."
        done
        if [ -n "$pip_install" ]; then
            cmd="python -m pip install $pip_install"
            eval $cmd
            exit_code=$?
            if [ $exit_code -ne 0 ]; then
                echo "python -m pip install pip_install failed with exit code $exit_code."
                self_delete $exit_code
            fi
        fi

        python -m pip install psycopg2-binary powerlift
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "python -m pip install psycopg2-binary powerlift failed with exit code $exit_code."
            self_delete $exit_code
        fi

        retry_count=0
        while true; do
            result=$(psql "$DB_URL" -c "SELECT script FROM Experiment WHERE id = '$EXPERIMENT_ID' LIMIT 1;" -t -A)
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                break
            fi
            echo "psql failed with exit code $exit_code."
            if [ $retry_count -ge 300 ]; then
                echo "Maximum number of retries reached. Command failed."
                self_delete $exit_code
            fi
            retry_count=$((retry_count + 1))
            echo "Sleeping."
            sleep 300
            echo "Retrying."
        done
        printf "%s" "$result" > "startup.py"

        while true; do
            echo "Running startup.py"
            python startup.py

            # 0 means we are done
            # 1 means we have more work
            # anything else is a serious error and we cleanup
            python_exit_code=$?
            echo "Powerlift startup.py script exited with code: $python_exit_code"

            if [ $python_exit_code -eq 0 ]; then
                break
            fi

            if [ $python_exit_code -ne 1 ]; then
                retry_count=0
                while true; do
                    result=$(psql "$DB_URL" -c "SELECT id FROM trial WHERE experiment_id = '$EXPERIMENT_ID' AND runner_id = '$RUNNER_ID' LIMIT 1;" -t -A)
                    exit_code=$?
                    if [ $exit_code -eq 0 ]; then
                        break
                    fi
                    echo "psql failed with exit code $exit_code."
                    if [ $retry_count -ge 300 ]; then
                        echo "Maximum number of retries reached. Command failed."
                        self_delete $python_exit_code
                    fi
                    retry_count=$((retry_count + 1))
                    echo "Sleeping."
                    sleep 300
                    echo "Retrying."
                done

                if [ -n "$result" ]; then
                    # Found an orphaned trial. Set the errmsg if it isn't already set.

                    retry_count=0
                    while true; do
                        psql "$DB_URL" -c "UPDATE trial SET runner_id = -2, end_time = CURRENT_TIMESTAMP, errmsg = 'ERROR: $python_exit_code' WHERE id = $result AND errmsg is NULL;"
                        exit_code=$?
                        if [ $exit_code -eq 0 ]; then
                            break
                        fi
                        echo "psql failed with exit code $exit_code."
                        if [ $retry_count -ge 300 ]; then
                            echo "Maximum number of retries reached. Command failed."
                            self_delete $python_exit_code
                        fi
                        retry_count=$((retry_count + 1))
                        echo "Sleeping."
                        sleep 300
                        echo "Retrying."
                    done
                fi
            fi
        done

        self_delete $python_exit_code
    """

    import time
    from heapq import heappush
    from datetime import datetime
    from azure.core.exceptions import HttpResponseError
    from azure.identity import ClientSecretCredential
    from azure.mgmt.containerinstance import ContainerInstanceManagementClient
    from azure.mgmt.containerinstance.models import (
        Container,
        ContainerGroup,
        ContainerGroupRestartPolicy,
        EnvironmentVariable,
        OperatingSystemTypes,
        ResourceRequests,
        ResourceRequirements,
    )
    from azure.mgmt.resource import ResourceManagementClient

    client_id = azure_json["client_id"]

    if credential is None:
        credential = ClientSecretCredential(
            tenant_id=azure_json["tenant_id"],
            client_id=client_id,
            client_secret=azure_json["client_secret"],
        )

    resource_group_name = azure_json["resource_group"]
    subscription_id = azure_json["subscription_id"]

    aci_client = None
    auth_client = None
    container_groups = []
    res_client = ResourceManagementClient(credential, subscription_id)

    # If this first call fails, then allow the Exception to propagate.
    resource_group_location = res_client.resource_groups.get(
        resource_group_name
    ).location

    container_resource_requests = ResourceRequests(
        cpu=num_cores,
        memory_in_gb=mem_size_gb,
    )
    container_resource_requirements = ResourceRequirements(
        requests=container_resource_requests
    )

    container_group_names = set()
    for runner_id in range(n_instances):
        container_group_name = f"powerlift-container-group-{batch_id}-{runner_id:04}"

        env_vars = [
            EnvironmentVariable(name="EXPERIMENT_ID", value=str(experiment_id)),
            EnvironmentVariable(name="RUNNER_ID", value=str(runner_id)),
            EnvironmentVariable(name="DB_URL", secure_value=uri),
            EnvironmentVariable(name="TIMEOUT", value=timeout),
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
            location=resource_group_location,
            containers=[container],
            os_type=OperatingSystemTypes.linux,
            restart_policy=ContainerGroupRestartPolicy.never,
            identity={"type": "SystemAssigned"},
        )

        if aci_client is None:
            aci_client = ContainerInstanceManagementClient(credential, subscription_id)

        max_attempts = 4
        while True:
            try:
                # begin_create_or_update returns LROPoller,
                # but this is only indicates when the container is started
                started = aci_client.container_groups.begin_create_or_update(
                    resource_group_name, container_group_name, container_group
                )
                break
            except HttpResponseError:
                max_attempts -= 1
                if max_attempts <= 0:
                    raise

        container_group_names.add(container_group_name)
        heappush(container_groups, (datetime.now(), container_group_name, started))
        aci_client, auth_client = assign_contributor_permissions(
            aci_client,
            auth_client,
            max_undead,
            credential,
            subscription_id,
            client_id,
            resource_group_name,
            resource_uris,
            container_groups,
        )

    assign_contributor_permissions(
        aci_client,
        auth_client,
        0,
        credential,
        subscription_id,
        client_id,
        resource_group_name,
        resource_uris,
        container_groups,
    )

    if delete_on_complete:
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
                except HttpResponseError:
                    pass

            for container_group_name in remove_after:
                container_group_names.remove(container_group_name)

            time.sleep(1)

        for deleted in deletes:
            while True:
                try:
                    while not deleted.done():
                        time.sleep(1)
                    break
                except HttpResponseError:
                    time.sleep(1)
