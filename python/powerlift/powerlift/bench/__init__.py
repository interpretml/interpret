"""Benchmarking module is responsible for building and running experiments.

The general philosophy is to allow
methods to be tested on both a single machine in the convenience of a notebook,
while also providing the same mechanisms across distributed platforms
when scale becomes a concern.

Key design considerations:
- API simplicity.
- Continual feedback to the user.
- User supplied code should be easily tested.
"""

from powerlift.bench.benchmark import Benchmark
from powerlift.bench.experiment import Experiment
from powerlift.bench.store import (
    DatasetAlreadyExistsError,
    Store,
    populate_with_datasets,
    retrieve_catboost_50k,
    retrieve_openml,
    retrieve_openml_automl_classification,
    retrieve_openml_automl_regression,
    retrieve_openml_cc18,
    retrieve_pmlb,
)
