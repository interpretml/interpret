""" Benchmarking module is responsible for building and running experiments.

The general philosophy is to allow
methods to be tested on both a single machine in the convenience of a notebook,
while also providing the same mechanisms across distributed platforms
when scale becomes a concern.

Key design considerations:
- API simplicity.
- Continual feedback to the user.
- User supplied code should be easily tested.
"""

from powerlift.bench.experiment import Experiment
from powerlift.bench.store import Store
from powerlift.bench.benchmark import Benchmark

from powerlift.bench.store import populate_with_datasets, DatasetAlreadyExistsError
from powerlift.bench.store import retrieve_openml, retrieve_pmlb, retrieve_catboost_50k
