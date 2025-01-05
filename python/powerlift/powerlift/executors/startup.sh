#!/bin/sh

self_delete() {
    echo "Attempt to self-delete this container group. Exit code was: $1"

    if [ $1 -ne 0 ]; then
        echo "Waiting 10 minutes to allow inspection of the logs..."
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