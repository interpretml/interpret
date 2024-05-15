# Powerlift

![License](https://img.shields.io/github/license/interpretml/interpret.svg?style=flat-square)
<br/>
> ### Advancing the state of machine learning?
> ### With 5-10 datasets? Wake me up when I'm dead.

Powerlift is all about testing machine learning techniques across many, many datasets. So many, that we had run into design of experiment concerns. So many, that we had to develop a package for it.

Yes, we run this for InterpretML on as many docker containers we can run in parallel on. Why wait days for benchmark evalations when you can wait for minutes? Rhetorical question, please don't hurt me.

```python
def trial_filter(task):
    if task.problem == "binary" and task.scalar_measure("n_rows") <= 10000:
        return ["rf", "svm"]
    return []

def trial_runner(trial):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

    if trial.task.problem == "binary" and trial.task.origin == "openml":
        X, y, meta = trial.task.data(["X", "y", "meta"])

        # Holdout split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3)

        # Build preprocessor
        is_cat = meta["categorical_mask"]
        cat_cols = [idx for idx in range(X.shape[1]) if is_cat[idx]]
        num_cols = [idx for idx in range(X.shape[1]) if not is_cat[idx]]
        cat_ohe_step = ("ohe", OneHotEncoder(sparse_output=True, handle_unknown="ignore"))
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
        if trial.method.name == "svm":
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

# Create experiment within database.
import os
from powerlift.bench import Benchmark
benchmark = Benchmark(f"sqlite:///{os.getcwd()}/powerlift.db", name="SVM vs RF")

# Only run this once for the database (downloads PMLB and OpenML CC18 datasets).
from powerlift.bench import populate_with_datasets
populate_with_datasets(benchmark.store, cache_dir="~/.powerlift")

# Run experiment
benchmark.run(trial_runner, trial_filter)
benchmark.wait_until_complete()
```

This can also be run on Azure Container Instances where needed.
```python
# Run experiment (but in ACI).
from powerlift.executors import AzureContainerInstance
azure_tenant_id = os.getenv("AZURE_TENANT_ID")
azure_client_id = os.getenv("AZURE_CLIENT_ID")
azure_client_secret = os.getenv("AZURE_CLIENT_SECRET")
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
store = Store(os.getenv("AZURE_DB_URL"))

executor = AzureContainerInstance(
    store,
    azure_tenant_id,
    azure_client_id,
    azure_client_secret,
    subscription_id,
    resource_group,
    n_running_containers=5,
    num_cores=1,
    mem_size_gb=2,
    raise_exception=True,
)
benchmark = Benchmark(store, name="SVM vs RF")
benchmark.run(trial_runner, trial_filter, timeout=10, executor=executor)
benchmark.wait_until_complete()
```

## Install
`pip install powerlift[datasets]`

That's it, go get 'em boss.