""" Dataset stores including utility methods. 

The end goal is to allow users to freely register their own benchmarks
for retrieval by their peers while also providing some basic benchmarks
for immediate testing.

Currently supported:
- PMLB
- OpenML CC18

Near future support:
- Ikonomovska regression datasets
- scikit-learn associated datasets
"""


import pytz
import base64
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Type
import random
import random
from powerlift.db.actions import delete_db, create_db
from powerlift.measures import class_stats, data_stats, regression_stats
from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine
from tqdm import tqdm
from itertools import chain
from sqlalchemy.orm import Session
import io
import os
from powerlift.db import schema as db
import numbers
from datetime import datetime
import pathlib
import pandas as pd
import ast


@dataclass
class Wheel:
    """Python wheel with its name and content as bytes."""

    name: str
    content: bytes


def _parse_function(src):
    src_ast = ast.parse(src)
    if isinstance(src_ast, ast.Module) and isinstance(src_ast.body[0], ast.FunctionDef):
        return src_ast
    return None


def _compile_function(src_ast):
    func_name = r"wired_function"
    src_ast.body[0].name = func_name
    compiled = compile(src_ast, "<string>", "exec")
    scope = locals()
    exec(compiled, scope, scope)
    return locals()[func_name]


MIMETYPE_DF = "application/vnd.interpretml/parquet-series"
MIMETYPE_SERIES = "application/vnd.interpretml/parquet-series"
MIMETYPE_JSON = "application/json"
MIMETYPE_PKL_FUNC = "application/vnd.interpretml/pickle-function"
MIMETYPE_WHEEL = "application/vnd.interpretml/python-wheel"


class BytesParser:
    @classmethod
    def deserialize(cls, mimetype, bytes):
        import io
        import json
        import pandas as pd

        bstream = io.BytesIO(bytes)
        if mimetype == MIMETYPE_JSON:
            return json.load(bstream)
        elif mimetype == MIMETYPE_DF:
            return pd.read_parquet(bstream)
        elif mimetype == MIMETYPE_SERIES:
            return pd.read_parquet(bstream)["Target"]
        elif mimetype == MIMETYPE_PKL_FUNC:
            src = bstream.getvalue().decode("utf-8")
            src_ast = _parse_function(src)
            if src_ast is None:
                raise RuntimeError("Serialized code not valid.")
            compiled_func = _compile_function(src_ast)
            return compiled_func
        elif mimetype == MIMETYPE_WHEEL:
            json_record = json.load(bstream)
            content = base64.b64decode(json_record["content"].encode("ascii"))
            return Wheel(json_record["name"], content)
        else:
            return None

    @classmethod
    def serialize(cls, obj):
        import io
        import json
        import pandas as pd
        from types import FunctionType
        import inspect

        bstream = io.BytesIO()
        mimetype = None
        if isinstance(obj, pd.Series):
            obj.astype(dtype=object).to_frame(name="Target").to_parquet(bstream)
            mimetype = MIMETYPE_SERIES
        elif isinstance(obj, pd.DataFrame):
            obj.to_parquet(bstream)
            mimetype = MIMETYPE_DF
        elif isinstance(obj, dict):
            bstream.write(json.dumps(obj).encode())
            mimetype = MIMETYPE_JSON
        elif isinstance(obj, FunctionType):
            src = inspect.getsource(obj)
            src_ast = _parse_function(src)
            if src_ast is None:
                raise RuntimeError("Serialized code not valid.")
            bstream.write(src.encode("utf-8"))
            mimetype = MIMETYPE_PKL_FUNC
        elif isinstance(obj, Wheel):
            content = base64.b64encode(obj.content).decode("ascii")
            json_record = {
                "name": obj.name,
                "content": content,
            }
            bstream.write(json.dumps(json_record).encode())
            mimetype = MIMETYPE_WHEEL
        else:
            return None, None
        return mimetype, bstream


class Store:
    """Store that represents persistent state for experiments.

    Apart from initialization, the user should not be using its methods normally.
    """

    def __init__(self, uri: str, force_recreate: bool = False, **create_engine_kwargs):
        """Initializes.

        Args:
            uri (str): Database URI to connect store to.
            force_recreate (bool, optional): This will delete and create the database associated with the uri if set to true. Defaults to False.
        """
        if force_recreate:
            delete_db(uri)
            self._engine = create_db(uri, **create_engine_kwargs)
        else:
            self._engine = create_engine(uri, **create_engine_kwargs)

        self._conn = self._engine.connect()
        self._session = Session(bind=self._conn)

        self._declared_measures = {}
        self._measure_counts = {}
        self._uri = uri

    @property
    def uri(self):
        return self._uri

    def __del__(self):
        self._session.close()
        self._conn.close()

    def start_trial(self, trial_id):
        trial_orm = self._session.query(db.Trial).filter_by(id=trial_id).one()
        start_time = datetime.now(pytz.utc)
        trial_orm.start_time = start_time
        trial_orm.status = db.StatusEnum.RUNNING
        self._session.add(trial_orm)
        self._session.commit()
        return start_time

    def end_trial(self, trial_id, errmsg=None):
        trial_orm = self._session.query(db.Trial).filter_by(id=trial_id).one()
        end_time = datetime.now(pytz.utc)
        trial_orm.end_time = end_time
        if errmsg is not None:
            trial_orm.errmsg = errmsg
            trial_orm.status = db.StatusEnum.ERROR
        else:
            trial_orm.status = db.StatusEnum.COMPLETE

        self._session.add(trial_orm)
        self._session.commit()
        return end_time

    def add_trial_run_fn(self, trial_ids, trial_run_fn, wheel_filepaths=None):
        import sys

        mimetype, bstream = BytesParser.serialize(trial_run_fn)
        trial_run_fn_asset_orm = db.Asset(
            name="trial_run_fn",
            description="Serialized trial run function.",
            version=sys.version,
            is_embedded=True,
            embedded=bstream.getvalue(),
            mimetype=mimetype,
        )
        wheel_asset_orms = []
        if wheel_filepaths is not None:
            for wheel_filepath in wheel_filepaths:
                with open(wheel_filepath, "rb") as f:
                    content = f.read()
                name = pathlib.Path(wheel_filepath).name
                wheel = Wheel(name, content)
                mimetype, bstream = BytesParser.serialize(wheel)
                wheel_asset_orm = db.Asset(
                    name=name,
                    description=f"Wheel: {name}",
                    version=sys.version,
                    is_embedded=True,
                    embedded=bstream.getvalue(),
                    mimetype=mimetype,
                )
                wheel_asset_orms.append(wheel_asset_orm)

        trial_orms = self._session.query(db.Trial).filter(db.Trial.id.in_(trial_ids))
        for trial_orm in trial_orms:
            trial_orm.input_assets.append(trial_run_fn_asset_orm)
            if len(wheel_asset_orms) > 0:
                trial_orm.input_assets.extend(wheel_asset_orms)

        if trial_orms.first() is not None:
            orms = [trial_run_fn_asset_orm]
            orms.extend(wheel_asset_orms)
            self._session.bulk_save_objects(orms, return_defaults=True)
            self._session.commit()
        return None

    def measure_from_db_task(self, task_orm):
        from powerlift.bench.experiment import Measure
        from collections import defaultdict

        measure_outcomes_orm = task_orm.measure_outcomes
        desc_id_to_measure_description_orm = {}
        desc_id_to_values = defaultdict(list)
        desc_name_to_measure = {}

        for measure_outcome_orm in measure_outcomes_orm:
            desc_id = measure_outcome_orm.measure_description_id
            if desc_id not in desc_id_to_measure_description_orm:
                measure_description_orm = measure_outcome_orm.measure_description
                desc_id_to_measure_description_orm[desc_id] = measure_description_orm
            measure_description_orm = desc_id_to_measure_description_orm[desc_id]
            _type = measure_description_orm.type
            if _type == db.TypeEnum.NUMBER:
                val = measure_outcome_orm.num_val
            elif _type == db.TypeEnum.STR:
                val = measure_outcome_orm.str_val
            elif _type == db.TypeEnum.JSON:
                val = measure_outcome_orm.json_val
            else:
                raise RuntimeError("Code branch should be unreachable")

            desc_id_to_values[desc_id].append(
                {
                    "seq_num": measure_outcome_orm.seq_num,
                    "timestamp": measure_outcome_orm.timestamp,
                    "val": val,
                }
            )

        for desc_id in desc_id_to_values.keys():
            values_df = pd.DataFrame.from_records(
                desc_id_to_values[desc_id], index="seq_num"
            )
            measure_description_orm = desc_id_to_measure_description_orm[desc_id]
            measure = Measure(
                measure_description_orm.name,
                measure_description_orm.description,
                measure_description_orm.type,
                measure_description_orm.lower_is_better,
                values_df,
            )
            desc_name_to_measure[measure.name] = measure
        return desc_name_to_measure

    def from_db_task(self, task_orm):
        from powerlift.bench.experiment import Task

        assets = [self.from_db_asset(asset) for asset in task_orm.assets]
        measures = self.measure_from_db_task(task_orm)
        return Task(
            task_orm.id,
            task_orm.name,
            task_orm.description,
            task_orm.version,
            task_orm.problem,
            task_orm.origin,
            task_orm.config,
            assets,
            measures,
        )

    def from_db_asset(self, asset_orm):
        from powerlift.bench.experiment import Asset

        return Asset(
            asset_orm.id,
            asset_orm.name,
            asset_orm.description,
            asset_orm.version,
            asset_orm.is_embedded,
            asset_orm.embedded,
            asset_orm.uri,
            asset_orm.mimetype,
        )

    def from_db_experiment(self, experiment_orm):
        from powerlift.bench.experiment import Experiment

        return Experiment(
            self, experiment_orm.name, experiment_orm.description, experiment_orm.id
        )

    def from_db_method(self, method_orm):
        from powerlift.bench.experiment import Method

        return Method(
            method_orm.id,
            method_orm.name,
            method_orm.description,
            method_orm.version,
            method_orm.params,
            method_orm.env,
        )

    def from_db_trial(self, trial_orm):
        from powerlift.bench.experiment import Trial

        experiment_orm = trial_orm.experiment
        experiment = self.from_db_experiment(experiment_orm)
        input_assets = [self.from_db_asset(asset) for asset in trial_orm.input_assets]
        task = self.from_db_task(trial_orm.task)
        method = self.from_db_method(trial_orm.method)
        return Trial(
            trial_orm.id,
            experiment,
            task,
            method,
            trial_orm.replicate_num,
            trial_orm.meta,
            input_assets,
        )

    def find_experiment_by_id(self, _id: int):
        experiment_orm = (
            self._session.query(db.Experiment).filter_by(id=_id).one_or_none()
        )
        if experiment_orm is None:
            return None
        return self.from_db_experiment(experiment_orm)

    def find_task_by_id(self, _id: int):
        task_orm = self._session.query(db.Task).filter_by(id=_id).one_or_none()
        if task_orm is None:
            return None
        return self.from_db_task(task_orm)

    def find_trial_by_id(self, _id: int):
        trial_orm = self._session.query(db.Trial).filter_by(id=_id).one_or_none()
        if trial_orm is None:
            return None
        return self.from_db_trial(trial_orm)

    def get_or_create_experiment(self, name: str, description: str) -> Tuple[int, bool]:
        """Get or create experiment keyed by name."""
        created = False
        exp_orm = self._session.query(db.Experiment).filter_by(name=name).one_or_none()
        if exp_orm is None:
            created = True
            exp_orm = db.Experiment(name=name, description=description)
            try:
                self._session.add(exp_orm)
                self._session.commit()
            except IntegrityError:
                self._session.rollback()
                exp_orm = self._session.query(db.Experiment).filter_by(name=name).one()
        return exp_orm.id, created

    def create_trials(self, trial_params: List[Dict[str, Any]]):
        trial_orms = []
        for trial_param in trial_params:
            trial_orm = db.Trial(
                status=db.StatusEnum.READY,
                create_time=datetime.now(pytz.utc),
                **trial_param,
            )
            trial_orms.append(trial_orm)
        self._session.bulk_save_objects(trial_orms, return_defaults=True)
        self._session.commit()
        return [x.id for x in trial_orms]

    def create_trial(
        self,
        experiment_id: int,
        task_id: int,
        method_id: int,
        replicate_num: int,
        meta: dict,
    ):
        trial_orm = db.Trial(
            experiment_id=experiment_id,
            task_id=task_id,
            method_id=method_id,
            replicate_num=replicate_num,
            meta=meta,
            status=db.StatusEnum.READY,
            create_time=datetime.now(pytz.utc),
        )
        self._session.add(trial_orm)
        self._session.commit()
        return trial_orm.id

    def get_or_create_method(
        self,
        name: str,
        description: str,
        version: str,
        params: dict,
        env: dict,
    ):
        """Get or create method keyed by name."""

        created = False
        method_orm = self._session.query(db.Method).filter_by(name=name).one_or_none()
        if method_orm is None:
            created = True
            method_orm = db.Method(
                name=name,
                description=description,
                version=version,
                params=params,
                env=env,
            )
            self._session.add(method_orm)
            self._session.commit()
        return method_orm.id, created

    def iter_tasks(self):
        for task_orm in self._session.query(db.Task).all():
            task = self.from_db_task(task_orm)
            yield task

    def add_measure(
        self,
        trial_or_task_id: int,
        trial_or_task_type: Type,
        name,
        value,
        description=None,
        type_=None,
        lower_is_better=True,
    ):
        from powerlift.bench.experiment import Task, Trial

        if type_ is None:
            if isinstance(value, str):
                type_ = db.TypeEnum.STR
            elif isinstance(value, dict):
                type_ = db.TypeEnum.JSON
            elif isinstance(value, numbers.Number):
                type_ = db.TypeEnum.NUMBER
            else:
                raise RuntimeError(
                    f"Value type {type(value)} is not supported for measure"
                )
        elif isinstance(type_, str):
            type_ = db.TypeEnum[type_.upper()]

        # Create measure description if needed
        is_declared = name in self._declared_measures
        if not is_declared:
            if description is None:
                description = f"Measure: {name}"

            measure_description_orm = (
                self._session.query(db.MeasureDescription)
                .filter_by(name=name)
                .one_or_none()
            )
            if measure_description_orm is None:
                measure_description_orm = db.MeasureDescription(
                    name=name,
                    description=description,
                    type=type_,
                    lower_is_better=lower_is_better,
                )
                try:
                    self._session.add(measure_description_orm)
                    self._session.commit()
                except IntegrityError:
                    self._session.rollback()
                    measure_description_orm = (
                        self._session.query(db.MeasureDescription)
                        .filter_by(name=name)
                        .one()
                    )
            self._declared_measures[name] = measure_description_orm
        else:
            measure_description_orm = self._declared_measures[name]

        # Create measure
        seq_num = self._measure_counts[name] = self._measure_counts.get(name, -1) + 1
        timestamp = datetime.now(pytz.utc)
        measure_outcome_orm = db.MeasureOutcome(
            measure_description=measure_description_orm,
            timestamp=timestamp,
            seq_num=seq_num,
        )
        if type_ == db.TypeEnum.STR:
            measure_outcome_orm.str_val = value
        elif type_ == db.TypeEnum.JSON:
            measure_outcome_orm.json_val = value
        elif type_ == db.TypeEnum.NUMBER:
            measure_outcome_orm.num_val = value
        else:
            raise RuntimeError(f"Value type {type(value)} is not supported for measure")

        if trial_or_task_type == Task:
            db_type = db.Task
        elif trial_or_task_type == Trial:
            db_type = db.Trial
        else:
            raise RuntimeError(f"Type {trial_or_task_type} is not Task nor Trial")

        trial_or_task_orm = (
            self._session.query(db_type).filter_by(id=trial_or_task_id).one()
        )
        trial_or_task_orm.measure_outcomes.append(measure_outcome_orm)
        if not is_declared:
            self._session.add(measure_description_orm)
        self._session.add(measure_outcome_orm)
        self._session.commit()
        return measure_outcome_orm.id

    def create_task_with_data(self, supervised, version="0.0.1"):
        X_bstream, y_bstream, meta_bstream = SupervisedDataset.serialize(supervised)
        X_name, y_name, meta_name = supervised.asset_names()
        X_mimetype, y_mimetype, meta_mimetype = supervised.mimetypes()

        X_orm = db.Asset(
            name=X_name,
            description=f"Training data for {supervised.name()}",
            version=version,
            is_embedded=True,
            mimetype=X_mimetype,
            embedded=X_bstream.getvalue(),
        )
        y_orm = db.Asset(
            name=y_name,
            description=f"Labels for {supervised.name()}",
            version=version,
            is_embedded=True,
            mimetype=y_mimetype,
            embedded=y_bstream.getvalue(),
        )

        meta_orm = db.Asset(
            name=meta_name,
            description=f"Metadata for {supervised.name()}",
            version=version,
            is_embedded=True,
            mimetype=meta_mimetype,
            embedded=meta_bstream.getvalue(),
        )

        meta = supervised.meta
        task_orm = db.Task(
            name=f"{meta['name']}:{meta['problem']}",
            description=f"Dataset {meta['name']} for {meta['problem']}",
            version=version,
            problem=meta["problem"],
            origin=meta["source"],
            config={
                "type": "data_supervised",
                "aliases": {
                    "X": X_name,
                    "y": y_name,
                    "meta": meta_name,
                },
            },
        )
        task_orm.assets.append(X_orm)
        task_orm.assets.append(y_orm)
        task_orm.assets.append(meta_orm)

        self._session.add(X_orm)
        self._session.add(y_orm)
        self._session.add(task_orm)
        self._session.commit()

        return task_orm.id


def populate_task_measures(store, task_id, supervised):
    from powerlift.bench.experiment import Task

    meta = supervised.meta
    unprocessed_measures = []
    if meta["problem"] == "regression":
        unprocessed_measures.extend(regression_stats(supervised.y))
    elif meta["problem"] in ["binary", "multiclass"]:
        unprocessed_measures.extend(class_stats(supervised.y))
    unprocessed_measures.extend(data_stats(supervised.X, meta["categorical_mask"]))

    for unprocessed_measure in unprocessed_measures:
        name, description, value, lower_is_better = unprocessed_measure
        store.add_measure(
            task_id,
            Task,
            name,
            value,
            description=description,
            lower_is_better=lower_is_better,
        )


def retrieve_cache(cache_dir: Optional[str], names: List[str]) -> List[io.BytesIO]:
    if cache_dir is None:
        return None

    cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name in names:
        filepath = pathlib.Path(cache_dir, name)
        if filepath.exists():
            with open(filepath, "rb") as f:
                outputs.append(io.BytesIO(f.read()))
        else:
            return None


def update_cache(cache_dir, names: List[str], bytes_io: List[io.BytesIO]):
    cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
    for i, _ in enumerate(names):
        filepath = pathlib.Path(cache_dir, names[i])
        with open(filepath, "wb") as f:
            f.write(bytes_io[i].getvalue())


@dataclass
class SupervisedDataset:
    X: pd.DataFrame
    y: pd.Series
    meta: dict

    @classmethod
    def serialize(cls, obj):
        _, X_bstream = BytesParser.serialize(obj.X)
        _, y_bstream = BytesParser.serialize(obj.y)
        _, meta_bstream = BytesParser.serialize(obj.meta)
        return X_bstream, y_bstream, meta_bstream

    @classmethod
    def deserialize(
        cls, X_bstream: io.BytesIO, y_bstream: io.BytesIO, meta_bstream: io.BytesIO
    ):
        X = BytesParser.deserialize(MIMETYPE_DF, X_bstream)
        y = BytesParser.deserialize(MIMETYPE_SERIES, y_bstream)
        meta = BytesParser.deserialize(MIMETYPE_JSON, meta_bstream)
        cls(X, y, meta)

    def asset_names(self):
        X_name = f"{self.meta['name']}.X.parquet"
        y_name = f"{self.meta['name']}.y.parquet"
        meta_name = f"{self.meta['name']}.meta.json"
        return X_name, y_name, meta_name

    def mimetypes(self):
        X_metadata = MIMETYPE_DF
        y_metadata = MIMETYPE_SERIES
        meta_metadata = MIMETYPE_JSON
        return X_metadata, y_metadata, meta_metadata

    def name(self):
        return self.meta["name"]


def populate_with_datasets(
    store: Store,
    dataset_iter: Iterable[SupervisedDataset] = None,
    cache_dir: str = None,
):
    """Populates store with datasets.

    Args:
        store (Store): Store for experiment.
        dataset_iter (Iterable[SupervisedDataset], optional): Iterable of supervised datasets. Defaults to None, which populates with OpenML and PMLB.
        cache_dir (str, optional): If dataset_iter is None, use this cache directory across calls. Defaults to None.
    """
    if dataset_iter is None:
        dataset_iter = chain(
            retrieve_openml(cache_dir=cache_dir), retrieve_pmlb(cache_dir=cache_dir)
        )

    for supervised in dataset_iter:
        task_id = store.create_task_with_data(supervised)
        populate_task_measures(store, task_id, supervised)


def retrieve_openml(cache_dir: str = None) -> Generator[SupervisedDataset]:
    """Retrives OpenML CC18 datasets.

    Args:
        cache_dir (str, optional): Use this cache directory across calls. Defaults to None.

    Yields:
        Generator[SupervisedDataset]: Yields datasets.
    """
    import openml

    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir, "openml")

    suite = openml.study.get_suite(99)
    tasks = suite.tasks.copy()
    random.Random(1337).shuffle(tasks)
    for task_id in tqdm(tasks, desc="openml"):
        task = openml.tasks.get_task(task_id)
        dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
        name = dataset.name
        X_name = f"{name}.X.parquet"
        y_name = f"{name}.y.parquet"
        meta_name = f"{name}.meta.json"
        problem = (
            "binary" if dataset.qualities["NumberOfClasses"] == 2 else "multiclass"
        )

        cached = retrieve_cache(cache_dir, [X_name, y_name, meta_name])
        if cached is None:
            X, y, categorical_mask, feature_names = task.get_dataset().get_data(
                target=task.target_name, dataset_format="dataframe"
            )
            meta = {
                "name": name,
                "problem": problem,
                "source": "openml",
                "categorical_mask": categorical_mask,
                "feature_names": feature_names,
            }
            supervised = SupervisedDataset(X, y, meta)
        else:
            supervised = SupervisedDataset.deserialize(*cached)

        if cache_dir is not None:
            serialized = SupervisedDataset.serialize(supervised)
            update_cache(cache_dir, [X_name, y_name, meta_name], serialized)
        yield supervised


def retrieve_pmlb(cache_dir: str = None) -> Generator[SupervisedDataset]:
    """Retrieves PMLB regression and classification datasets.

    Args:
        cache_dir (str, optional): Use this cache directory across calls. Defaults to None.

    Yields:
        Generator[SupervisedDataset]: Yields datasets.
    """
    from pmlb import fetch_data, classification_dataset_names, regression_dataset_names

    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir, "pmlb")

    dataset_names = []
    dataset_names.extend(
        [("classification", name) for name in classification_dataset_names]
    )
    dataset_names.extend([("regression", name) for name in regression_dataset_names])

    for problem_type, dataset_name in tqdm(dataset_names, desc="pmlb"):
        name = dataset_name
        X_name = f"{name}.X.parquet"
        y_name = f"{name}.y.parquet"
        meta_name = f"{name}.meta.json"

        cached = retrieve_cache(cache_dir, [X_name, y_name, meta_name])
        if cached is None:
            df = fetch_data(dataset_name)
            X = df.drop("target", axis=1)
            y = df["target"]
            problem = problem_type
            if problem_type == "classification":
                problem = "binary" if len(y.unique()) == 2 else "multiclass"
            meta = {
                "name": name,
                "problem": problem,
                "source": "pmlb",
                "categorical_mask": [dt.kind == "O" for dt in X.dtypes],
                "feature_names": list(X.columns),
            }
            supervised = SupervisedDataset(X, y, meta)
        else:
            supervised = SupervisedDataset.deserialize(*cached)

        if cache_dir is not None:
            serialized = SupervisedDataset.serialize(supervised)
            update_cache(cache_dir, [X_name, y_name, meta_name], serialized)
        yield supervised
