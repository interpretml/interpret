"""This is called to run a trial by worker nodes for Azure VMs."""


def assign_contributor_permissions(
    compute_client,
    auth_client,
    max_undead,
    credential,
    subscription_id,
    client_id,
    resource_group_name,
    resource_uris,
    vms,
):
    from heapq import heappush, heappop
    from datetime import datetime
    import time
    import uuid
    from azure.mgmt.compute import ComputeManagementClient
    from azure.mgmt.authorization import AuthorizationManagementClient
    from azure.mgmt.authorization.models import RoleAssignmentCreateParameters
    from azure.core.exceptions import HttpResponseError

    # Contributor Role
    contributor_definition_id = f"/subscriptions/{subscription_id}/providers/Microsoft.Authorization/roleDefinitions/b24988ac-6180-42a0-ab88-20f7382dd24c"

    while max_undead < len(vms):
        _, vm_name, started = heappop(vms)
        try:
            if started is not None:
                if not started.done():
                    heappush(
                        vms,
                        (datetime.now(), vm_name, started),
                    )
                    time.sleep(1)
                    continue
                started = None

            if compute_client is None:
                compute_client = ComputeManagementClient(credential, subscription_id)

            vm = compute_client.virtual_machines.get(resource_group_name, vm_name)

            ra_user_contributor = RoleAssignmentCreateParameters(
                role_definition_id=contributor_definition_id,
                principal_id=vm.identity.principal_id,
                principal_type="ServicePrincipal",
            )
            ra_principal_contributor = RoleAssignmentCreateParameters(
                role_definition_id=contributor_definition_id,
                principal_id=client_id,
                principal_type="User",
            )
            scope = f"/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.Compute/virtualMachines/{vm_name}"

            if auth_client is None:
                auth_client = AuthorizationManagementClient(credential, subscription_id)

            auth_client.role_assignments.create(
                scope, str(uuid.uuid4()), ra_user_contributor
            )
            if resource_uris is not None:
                for resource_uri in resource_uris:
                    auth_client.role_assignments.create(
                        resource_uri, str(uuid.uuid4()), ra_user_contributor
                    )
            auth_client.role_assignments.create(
                scope, str(uuid.uuid4()), ra_principal_contributor
            )
        except HttpResponseError:
            compute_client = None
            auth_client = None
            heappush(vms, (datetime.now(), vm_name, started))
            time.sleep(1)

    return compute_client, auth_client


def run_azure_process(
    experiment_id,
    n_instances,
    uri,
    resource_uris,
    timeout,
    azure_json,
    credential,
    location,
    vm_size,
    image_publisher,
    image_offer,
    image_sku,
    image_version,
    disk_type,
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
            VM_NAME=${VM_NAME}

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
                az vm delete --subscription $SUBSCRIPTION_ID --resource-group $RESOURCE_GROUP_NAME --name $VM_NAME --yes
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
    
        apt-get --yes update
        apt-get --yes install python3 python3-pip python3-venv postgresql-client
        python3 -m venv powerenv
        source powerenv/bin/activate
        python3 -m pip install --upgrade pip setuptools wheel packaging

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
                cmd="python3 -m pip install $filename"
                eval $cmd
                exit_code=$?
                if [ $exit_code -ne 0 ]; then
                    echo "python3 -m pip install filename failed with exit code $exit_code."
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
            cmd="python3 -m pip install $pip_install"
            eval $cmd
            exit_code=$?
            if [ $exit_code -ne 0 ]; then
                echo "python3 -m pip install pip_install failed with exit code $exit_code."
                self_delete $exit_code
            fi
        fi

        python3 -m pip install psycopg2-binary powerlift
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "python3 -m pip install psycopg2-binary powerlift failed with exit code $exit_code."
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
            python3 startup.py

            # 0 means we are done
            # 1 means we have more work
            # anything else is a serious error and we cleanup
            python_exit_code=$?
            echo "Powerlift startup.py script exited with code: $python_exit_code"

            if [ $python_exit_code -ne 1 ]; then
                break
            fi
        done
        
        if [ $python_exit_code -ne 0 ]; then
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

        self_delete $python_exit_code
    """

    import time
    from heapq import heappush
    from datetime import datetime
    from azure.core.exceptions import HttpResponseError
    from azure.identity import ClientSecretCredential
    from azure.mgmt.resource import ResourceManagementClient
    from azure.mgmt.compute import ComputeManagementClient
    from azure.mgmt.network import NetworkManagementClient
    from azure.mgmt.compute.models import DiskCreateOptionTypes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    import base64

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    public_key = (
        private_key.public_key()
        .public_bytes(
            serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH
        )
        .decode("utf-8")
    )

    client_id = azure_json["client_id"]

    if credential is None:
        credential = ClientSecretCredential(
            tenant_id=azure_json["tenant_id"],
            client_id=client_id,
            client_secret=azure_json["client_secret"],
        )

    resource_group_name = azure_json["resource_group"]
    subscription_id = azure_json["subscription_id"]

    if location is None:
        res_client = ResourceManagementClient(credential, subscription_id)
        location = res_client.resource_groups.get(resource_group_name).location

    compute_client = ComputeManagementClient(credential, subscription_id)
    network_client = NetworkManagementClient(credential, subscription_id)

    vnet_name = f"powerlift-vnet-{location}"
    subnet_name = f"powerlift-subnet-{location}"

    async_vnet_creation = network_client.virtual_networks.begin_create_or_update(
        resource_group_name,
        vnet_name,
        {"location": location, "address_space": {"address_prefixes": ["10.0.0.0/16"]}},
    )
    vnet = async_vnet_creation.result()

    async_subnet_creation = network_client.subnets.begin_create_or_update(
        resource_group_name, vnet_name, subnet_name, {"address_prefix": "10.0.0.0/24"}
    )
    subnet = async_subnet_creation.result()

    auth_client = None
    vms = []
    for runner_id in range(n_instances):
        nic_name = f"powerlift-nic-{batch_id}-{runner_id:04}"
        vm_name = f"powerlift-vm-{batch_id}-{runner_id:04}"

        async_nic_creation = network_client.network_interfaces.begin_create_or_update(
            resource_group_name,
            nic_name,
            {
                "location": location,
                "ip_configurations": [
                    {
                        "name": "ipConfig1",
                        "subnet": {"id": subnet.id},
                    }
                ],
                "delete_option": "Delete",
            },
        )
        nic = async_nic_creation.result()

        admin_username = "azureuser"

        full_startup_script = f"""#!/bin/bash
export EXPERIMENT_ID="{experiment_id}"
export RUNNER_ID="{runner_id}"
export DB_URL="{uri}"
export TIMEOUT="{timeout}"
export RESOURCE_GROUP_NAME="{resource_group_name}"
export VM_NAME="{vm_name}"
export SUBSCRIPTION_ID="{subscription_id}"
{startup_script}  
"""

        encoded_script = base64.b64encode(
            full_startup_script.replace("\r\n", "\n").encode("utf-8")
        ).decode("utf-8")

        vm_parameters = {
            "location": location,
            "os_profile": {
                "computer_name": vm_name,
                "admin_username": admin_username,
                "linux_configuration": {
                    "disable_password_authentication": True,
                    "ssh": {
                        "public_keys": [
                            {
                                "path": f"/home/{admin_username}/.ssh/authorized_keys",
                                "key_data": public_key,
                            }
                        ]
                    },
                },
                "custom_data": encoded_script,
            },
            "hardware_profile": {"vm_size": vm_size},
            "storage_profile": {
                "image_reference": {
                    "publisher": image_publisher,
                    "offer": image_offer,
                    "sku": image_sku,
                    "version": image_version,
                },
                "os_disk": {
                    "create_option": DiskCreateOptionTypes.FROM_IMAGE,
                    "managed_disk": {"storage_account_type": disk_type},
                    "name": f"{vm_name}_osdisk",
                    "delete_option": "Delete",
                },
            },
            "network_profile": {
                "network_interfaces": [
                    {
                        "id": nic.id,
                        "primary": True,
                        "properties": {"deleteOption": "Delete"},
                    }
                ]
            },
            "identity": {"type": "SystemAssigned"},
        }

        while True:
            try:
                # begin_create_or_update returns LROPoller,
                # but this is only indicates when the containter is started
                started = compute_client.virtual_machines.begin_create_or_update(
                    resource_group_name, vm_name, vm_parameters
                )
                break
            except HttpResponseError:
                time.sleep(1)

        heappush(vms, (datetime.now(), vm_name, started))
        compute_client, auth_client = assign_contributor_permissions(
            compute_client,
            auth_client,
            max_undead,
            credential,
            subscription_id,
            client_id,
            resource_group_name,
            resource_uris,
            vms,
        )

    assign_contributor_permissions(
        compute_client,
        auth_client,
        0,
        credential,
        subscription_id,
        client_id,
        resource_group_name,
        resource_uris,
        vms,
    )
