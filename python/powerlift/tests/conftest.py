from itertools import islice
from powerlift.bench.store import (
    Store,
    populate_with_datasets,
    retrieve_openml,
    retrieve_pmlb,
)
import pytest
import os


@pytest.fixture(scope="session")
def cache_dir():
    yield "~/.powerlift/test_cache"


@pytest.fixture(scope="session")
def dataset_limit():
    yield 5


@pytest.fixture(scope="session")
def uri():
    pw = os.environ.get("TEST_DB_PASS", None)
    if pw:
        yield f"postgresql://postgres:{pw}@localhost/test_powerlift"
    else:
        yield "postgresql://localhost/test_powerlift"


@pytest.fixture(scope="session")
def populated_uri():
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
    yield store


@pytest.fixture(scope="session")
def populated_azure_uri():
    import os

    yield os.getenv("AZURE_DB_URL")


@pytest.fixture(scope="session")
def populated_azure_store(populated_azure_uri, dataset_limit):
    store = Store(populated_azure_uri, force_recreate=True)
    dataset_iter = islice(retrieve_pmlb(), dataset_limit)
    populate_with_datasets(store, dataset_iter)
    yield store


@pytest.fixture(scope="session")
def store(uri):
    yield Store(uri, force_recreate=True)
    # delete_db(uri)
