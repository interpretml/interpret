from powerlift.bench import Store, Benchmark

from powerlift.executors.docker import InsecureDocker
from powerlift.executors.localmachine import LocalMachine
from powerlift.executors.azure_ci import AzureContainerInstance
import pytest
import os


def _add(x, y):
    return x + y


def _err_handler(e):
    raise e


def _trials(task):
    if task.problem == "binary" and task.scalar_measure("n_rows") <= 10000:
        return ["rf", "svm"]
    return []


def _benchmark(trial):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

    if trial.task.problem == "binary" and trial.task.origin in [
        "openml",
        "pmlb",
        "catboost_50k",
    ]:
        X, y, meta = trial.task.data(["X", "y", "meta"])

        # Holdout split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3)

        # Build preprocessor
        is_cat = meta["categorical_mask"]
        cat_cols = [idx for idx in range(X.shape[1]) if is_cat[idx]]
        num_cols = [idx for idx in range(X.shape[1]) if not is_cat[idx]]
        cat_ohe_step = (
            "ohe",
            OneHotEncoder(sparse_output=True, handle_unknown="ignore"),
        )
        cat_pipe = Pipeline([cat_ohe_step])
        num_pipe = Pipeline([("identity", FunctionTransformer())])
        transformers = [("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)]
        ct = Pipeline(
            [
                ("ct", ColumnTransformer(transformers=transformers)),
                (
                    "missing",
                    SimpleImputer(add_indicator=True, strategy="most_frequent"),
                ),
            ]
        )
        # Connect preprocessor with target learner
        if trial.method == "svm":
            clf = Pipeline([("ct", ct), ("est", CalibratedClassifierCV(LinearSVC()))])
        else:
            clf = Pipeline([("ct", ct), ("est", RandomForestClassifier())])

        # Train
        clf.fit(X_tr, y_tr)

        # Predict
        predictions = clf.predict_proba(X_te)[:, 1]

        # Score
        auc = roc_auc_score(y_te, predictions)
        trial.log("auc", auc)


def _assert_benchmark(benchmark):
    experiment = benchmark._experiment()
    assert len(experiment.trials) > 0
    status = benchmark.status()
    assert len(status) > 0
    results = benchmark.results()
    assert len(results) > 0
    available_tasks = benchmark.available_tasks(include_measures=True)
    assert len(available_tasks) > 0


@pytest.mark.skip("Failing.")
def test_scikit_experiment_debug(populated_store):
    store = populated_store
    executor = LocalMachine(store, n_cpus=1, debug_mode=True)
    benchmark = Benchmark(store, "scikit_debug")
    benchmark.run(_benchmark, _trials, timeout=60 * 10, executor=executor)
    benchmark.wait_until_complete()
    _assert_benchmark(benchmark)


@pytest.mark.skip("Failing.")
def test_scikit_experiment_local(populated_store):
    store = populated_store
    executor = LocalMachine(store, n_cpus=2)
    benchmark = Benchmark(store, name="scikit")
    benchmark.run(_benchmark, _trials, timeout=10, executor=executor)
    benchmark.wait_until_complete()
    _assert_benchmark(benchmark)


@pytest.mark.skip("Enable this when testing docker.")
def test_scikit_experiment_docker(populated_docker_store, populated_docker_uri):
    executor = InsecureDocker(
        populated_docker_store,
        n_running_containers=2,
        docker_db_uri=populated_docker_uri,
    )
    benchmark = Benchmark(populated_docker_store, name="scikit_docker")
    benchmark.run(_benchmark, _trials, timeout=60, executor=executor)
    benchmark.wait_until_complete()

    _assert_benchmark(benchmark)


@pytest.mark.skip("Failing.")
def test_scikit_experiment_aci(populated_azure_store):
    """
    As of 2022-06-09:
    - Takes roughly 20 seconds to submit 10 tasks.
    - Roughly 80 seconds for first runs to return.
    - 180 seconds to complete (5 parallel containers).
    """
    from dotenv import load_dotenv

    load_dotenv()
    azure_tenant_id = os.getenv("AZURE_TENANT_ID")
    azure_client_id = os.getenv("AZURE_CLIENT_ID")
    azure_client_secret = os.getenv("AZURE_CLIENT_SECRET")
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")

    store = populated_azure_store
    executor = AzureContainerInstance(
        store,
        azure_tenant_id,
        subscription_id,
        azure_client_id,
        azure_client_secret=azure_client_secret,
        resource_group=resource_group,
        n_running_containers=5,
        num_cores=2,
        mem_size_gb=8,
        delete_group_container_on_complete=False,
    )
    benchmark = Benchmark(store, name="scikit")
    benchmark.run(_benchmark, _trials, timeout=60, executor=executor)
    benchmark.wait_until_complete()
    _assert_benchmark(benchmark)


def test_multiprocessing():
    """This tests exists to ensure there is no hang in pytest."""
    from multiprocessing.pool import Pool

    pool = Pool()
    results = []
    num_tasks = 32
    for i in range(num_tasks):
        result = pool.apply_async(_add, (i, i), error_callback=_err_handler)
        results.append(result)
    counter = 0
    for i in range(num_tasks):
        counter += results[i].get()
    assert counter == 992
    pool.close()
