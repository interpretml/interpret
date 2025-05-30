import os
from itertools import islice

import pytest
from powerlift.bench.store import (
    Store,
    populate_with_datasets,
    retrieve_openml,
    retrieve_pmlb,
)


@pytest.fixture(scope="session")
def cache_dir():
    return "~/.powerlift/test_cache"


@pytest.fixture(scope="session")
def dataset_limit():
    return 5


@pytest.fixture(scope="session")
def uri():
    from dotenv import load_dotenv

    load_dotenv()

    pw = os.environ.get("TEST_DB_PASS", None)
    if pw:
        yield f"postgresql://postgres:{pw}@localhost/test_powerlift"
    else:
        yield "postgresql://localhost/test_powerlift"


@pytest.fixture(scope="session")
def populated_uri():
    from dotenv import load_dotenv

    load_dotenv()

    pw = os.environ.get("TEST_DB_PASS", None)
    if pw:
        yield f"postgresql://postgres:{pw}@localhost/test_powerlift_populated"
    else:
        yield "postgresql://localhost/test_powerlift_populated"


@pytest.fixture(scope="session")
def populated_store(populated_uri, dataset_limit):
    store = Store(populated_uri, force_recreate=True)
    dataset_iter = islice(retrieve_openml(), dataset_limit)
    populate_with_datasets(store, dataset_iter)
    return store


@pytest.fixture(scope="session")
def populated_azure_uri():
    import os

    from dotenv import load_dotenv

    load_dotenv()

    return os.getenv("AZURE_DB_URL")


@pytest.fixture(scope="session")
def populated_docker_uri():
    import os

    from dotenv import load_dotenv

    load_dotenv()

    return os.getenv("DOCKER_DB_URL")


@pytest.fixture(scope="session")
def populated_azure_store(populated_azure_uri, dataset_limit):
    store = Store(populated_azure_uri, force_recreate=True)
    dataset_iter = islice(retrieve_pmlb(), dataset_limit)
    populate_with_datasets(store, dataset_iter)
    return store


@pytest.fixture(scope="session")
def populated_docker_store(populated_docker_uri, dataset_limit):
    store = Store(populated_docker_uri, force_recreate=True)
    dataset_iter = islice(retrieve_pmlb(), dataset_limit)
    populate_with_datasets(store, dataset_iter)
    return store


@pytest.fixture(scope="session")
def store(uri):
    return Store(uri, force_recreate=True)
    # delete_db(uri)
