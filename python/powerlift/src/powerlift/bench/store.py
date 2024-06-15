""" Dataset stores including utility methods. 

The end goal is to allow users to freely register their own benchmarks
for retrieval by their peers while also providing some basic benchmarks
for immediate testing.

Currently supported:
- PMLB
- OpenML CC18
- CatBoost (regression/classification for <50k instances)

Near future support:
- Ikonomovska regression datasets
- scikit-learn associated datasets

# TODO(nopdive): Review how seq_num (integrity) are done with measure outcomes.
"""

import pytz
import base64
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Type, Mapping
import random
import random
from powerlift.db.actions import drop_tables, create_db, create_tables
from powerlift.measures import class_stats, data_stats, regression_stats
from sqlalchemy.exc import IntegrityError
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
MIMETYPE_FUNC = "application/vnd.interpretml/function-str"
MIMETYPE_WHEEL = "application/vnd.interpretml/python-wheel"


class BytesParser:
    @classmethod
    def deserialize(cls, mimetype, bytes):
        import io
        import json
        import pandas as pd

        if not isinstance(bytes, io.BytesIO):
            bstream = io.BytesIO(bytes)
        else:
            bstream = bytes

        if mimetype == MIMETYPE_JSON:
            return json.load(bstream)
        elif mimetype == MIMETYPE_DF:
            return pd.read_parquet(bstream)
        elif mimetype == MIMETYPE_SERIES:
            return pd.read_parquet(bstream)["Target"]
        elif mimetype == MIMETYPE_FUNC:
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
            orig_close = bstream.close
            bstream.close = lambda: None
            try:
                obj.astype(dtype=object).to_frame(name="Target").to_parquet(bstream)
            finally:
                bstream.close = orig_close
            mimetype = MIMETYPE_SERIES
        elif isinstance(obj, pd.DataFrame):
            orig_close = bstream.close
            bstream.close = lambda: None
            try:
                obj.to_parquet(bstream)
            finally:
                bstream.close = orig_close
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
            mimetype = MIMETYPE_FUNC
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
        self._engine = create_db(uri, **create_engine_kwargs)
        if force_recreate:
            drop_tables(self._engine)
        create_tables(self._engine)

        self._conn = self._engine.connect()
        self._session = Session(bind=self._conn)

        self._declared_measures_cache = {}
        self._measure_counts = {}
        self._uri = uri

    @property
    def uri(self):
        return self._uri

    def __del__(self):
        self._session.close()
        self._conn.close()

    def rollback(self):
        self._session.rollback()

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

        input_assets = [self.from_db_asset(asset) for asset in trial_orm.input_assets]
        task = self.from_db_task(trial_orm.task)
        method = self.from_db_method(trial_orm.method)
        return Trial(
            trial_orm.id,
            self,
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

    def get_experiment(self, name: str) -> Optional[int]:
        exp_orm = self._session.query(db.Experiment).filter_by(name=name).one_or_none()
        if exp_orm is None:
            return None
        else:
            return exp_orm.id

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

    def iter_experiment_trials(self, experiment_id: int):
        trial_orms = self._session.query(db.Trial).filter_by(
            experiment_id=experiment_id
        )
        for trial_orm in trial_orms:
            trial = self.from_db_trial(trial_orm)
            yield trial

    def iter_status(self, experiment_id: int) -> Iterable[Mapping[str, object]]:
        # TODO(nopdive): Should this be in the store?
        trial_orms = self._session.query(db.Trial).filter_by(
            experiment_id=experiment_id
        )
        for trial_orm in trial_orms:
            record = {
                "trial_id": trial_orm.id,
                "replicate_num": trial_orm.replicate_num,
                "meta": trial_orm.meta,
                "method": trial_orm.method.name,
                "task": trial_orm.task.name,
                "status": trial_orm.status.name,
                "errmsg": trial_orm.errmsg,
                "create_time": trial_orm.create_time,
                "start_time": trial_orm.start_time,
                "end_time": trial_orm.end_time,
            }
            yield record

    def iter_results(self, experiment_id: int) -> Iterable[Mapping[str, object]]:
        trial_orms = self._session.query(db.Trial).filter_by(
            experiment_id=experiment_id
        )
        for trial_orm in trial_orms:
            for measure_outcome in trial_orm.measure_outcomes:
                record = {
                    "trial_id": trial_orm.id,
                    "replicate_num": trial_orm.replicate_num,
                    "meta": trial_orm.meta,
                    "method": trial_orm.method.name,
                    "task": trial_orm.task.name,
                    "name": measure_outcome.measure_description.name,
                    "seq_num": measure_outcome.seq_num,
                    "type": measure_outcome.measure_description.type.name,
                    "num_val": measure_outcome.num_val,
                    "str_val": measure_outcome.str_val,
                    "json_val": measure_outcome.json_val,
                }
                yield record

    def iter_available_tasks(
        self, include_measures: bool = False
    ) -> Iterable[Mapping[str, object]]:
        task_orms = self._session.query(db.Task)
        for task_orm in task_orms:
            record = {
                "task_id": task_orm.id,
                "name": task_orm.name,
                "version": task_orm.version,
                "problem": task_orm.problem,
                "origin": task_orm.origin,
                "config": task_orm.config,
            }
            if include_measures:
                for measure_outcome in task_orm.measure_outcomes:
                    record.update(
                        {
                            "measure_name": measure_outcome.measure_description.name,
                            "seq_num": measure_outcome.seq_num,
                            "type": measure_outcome.measure_description.type.name,
                            "num_val": measure_outcome.num_val,
                            "str_val": measure_outcome.str_val,
                            "json_val": measure_outcome.json_val,
                        }
                    )
            yield record

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
        is_declared = name in self._declared_measures_cache
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
            self._declared_measures_cache[name] = measure_description_orm
        else:
            measure_description_orm = self._declared_measures_cache[name]

        # Create measure
        seq_num = self._measure_counts[name] = self._measure_counts.get(name, -1) + 1
        timestamp = datetime.now(pytz.utc)
        measure_outcome_orm = db.MeasureOutcome(
            measure_description=measure_description_orm,
            timestamp=timestamp,
            seq_num=seq_num,
        )
        self._session.add(measure_outcome_orm)

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
        self._session.commit()
        return measure_outcome_orm.id

    def create_task_with_data(self, data, version="0.0.1"):
        if isinstance(data, SupervisedDataset):
            return self._create_task_with_supervised(data, version)
        elif isinstance(data, DataFrameDataset):
            return self._create_task_with_dataframe(data, version)
        else:  # pragma: no cover
            raise ValueError(f"Does not support {type(data)}")

    def _create_task_with_supervised(self, supervised, version):
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
            name=meta["name"],
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

    def _create_task_with_dataframe(self, data, version):
        inputs_bstream, outputs_bstream, meta_bstream = DataFrameDataset.serialize(data)
        inputs_name, outputs_name, meta_name = data.asset_names()
        inputs_mimetype, outputs_mimetype, meta_mimetype = data.mimetypes()

        inputs_orm = db.Asset(
            name=inputs_name,
            description=f"Inputs for {data.name()}",
            version=version,
            is_embedded=True,
            mimetype=inputs_mimetype,
            embedded=inputs_bstream.getvalue(),
        )
        outputs_orm = db.Asset(
            name=outputs_name,
            description=f"Outputs for {data.name()}",
            version=version,
            is_embedded=True,
            mimetype=outputs_mimetype,
            embedded=outputs_bstream.getvalue(),
        )

        meta_orm = db.Asset(
            name=meta_name,
            description=f"Metadata for {data.name()}",
            version=version,
            is_embedded=True,
            mimetype=meta_mimetype,
            embedded=meta_bstream.getvalue(),
        )

        meta = data.meta
        task_orm = db.Task(
            name=meta["name"],
            description=f"Dataset {meta['name']} for {meta['problem']}",
            version=version,
            problem=meta["problem"],
            origin=meta["source"],
            config={
                "type": "data_dataframe",
                "aliases": {
                    "inputs": inputs_name,
                    "outputs": outputs_name,
                    "meta": meta_name,
                },
            },
        )
        task_orm.assets.append(inputs_orm)
        task_orm.assets.append(outputs_orm)
        task_orm.assets.append(meta_orm)

        self._session.add(inputs_orm)
        self._session.add(outputs_orm)
        self._session.add(task_orm)
        self._session.commit()

        return task_orm.id


def populate_task_measures(store, task_id, data):
    from powerlift.bench.experiment import Task

    meta = data.meta
    unprocessed_measures = []
    if isinstance(data, SupervisedDataset):
        if meta["problem"] == "regression":
            unprocessed_measures.extend(regression_stats(data.y))
        elif meta["problem"] in ["binary", "multiclass"]:
            unprocessed_measures.extend(class_stats(data.y))
        unprocessed_measures.extend(data_stats(data.X, meta["categorical_mask"]))
    elif isinstance(data, DataFrameDataset):
        inputs_data_stats = [
            (f"inputs_{x1}", x2, x3, x4)
            for x1, x2, x3, x4 in data_stats(
                data.inputs, meta["inputs_categorical_mask"]
            )
        ]
        outputs_data_stats = [
            (f"outputs_{x1}", x2, x3, x4)
            for x1, x2, x3, x4 in data_stats(
                data.outputs, meta["outputs_categorical_mask"]
            )
        ]
        unprocessed_measures.extend(inputs_data_stats)
        unprocessed_measures.extend(outputs_data_stats)

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


def retrieve_cache(
    cache_dir: Optional[str], names: List[str]
) -> Optional[List[io.BytesIO]]:
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
    return outputs


def update_cache(cache_dir, names: List[str], bytes_io: List[io.BytesIO]):
    cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
    for name, a_bytes_io in zip(names, bytes_io):
        filepath = pathlib.Path(cache_dir, name)
        with open(filepath, "wb") as f:
            b = a_bytes_io.getvalue()
            f.write(b)


class Dataset:
    pass


@dataclass
class DataFrameDataset(Dataset):
    inputs: pd.DataFrame
    outputs: pd.DataFrame
    meta: dict

    @classmethod
    def serialize(cls, obj):
        _, inputs_bstream = BytesParser.serialize(obj.inputs)
        _, outputs_bstream = BytesParser.serialize(obj.outputs)
        _, meta_bstream = BytesParser.serialize(obj.meta)
        return inputs_bstream, outputs_bstream, meta_bstream

    @classmethod
    def deserialize(
        cls,
        inputs_bstream: io.BytesIO,
        outputs_bstream: io.BytesIO,
        meta_bstream: io.BytesIO,
    ):
        inputs = BytesParser.deserialize(MIMETYPE_DF, inputs_bstream)
        outputs = BytesParser.deserialize(MIMETYPE_DF, outputs_bstream)
        meta = BytesParser.deserialize(MIMETYPE_JSON, meta_bstream)
        return cls(inputs, outputs, meta)

    def asset_names(self):
        inputs_name = f"{self.meta['name']}.inputs.parquet"
        outputs_name = f"{self.meta['name']}.outputs.parquet"
        meta_name = f"{self.meta['name']}.meta.json"
        return inputs_name, outputs_name, meta_name

    def mimetypes(self):
        inputs_metadata = MIMETYPE_DF
        outputs_metadata = MIMETYPE_DF
        meta_metadata = MIMETYPE_JSON
        return inputs_metadata, outputs_metadata, meta_metadata

    def name(self):
        return self.meta["name"]


@dataclass
class SupervisedDataset(Dataset):
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
        return cls(X, y, meta)

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


class DatasetAlreadyExistsError(Exception):
    """Raised when dataset already exists in store."""
    pass

def populate_with_datasets(
    store: Store,
    dataset_iter: Iterable[Dataset] = None,
    cache_dir: str = None,
    exist_ok: bool = False,
) -> bool:
    """Populates store with datasets.

    Attempts to add datasets to store from iterable.
    This can be made idempotent by setting `exist_ok` to true.

    Args:
        store (Store): Store for experiment.
        dataset_iter (Iterable[Dataset], optional): Iterable of supervised datasets. Defaults to None, which populates with OpenML and PMLB.
        cache_dir (str, optional): If dataset_iter is None, use this cache directory across calls. Defaults to None.
        exist_ok (bool, optional): Do not raise exception if a dataset already exist.
    Returns:
        bool: True if datasets were created, False otherwise
    """

    if dataset_iter is None:
        dataset_iter = chain(
            retrieve_openml(cache_dir=cache_dir), retrieve_pmlb(cache_dir=cache_dir)
        )

    for dataset in dataset_iter:
        try:
            task_id = store.create_task_with_data(dataset)
            populate_task_measures(store, task_id, dataset)
        except IntegrityError as e:
            store.rollback()
            if not exist_ok:
                raise DatasetAlreadyExistsError("Dataset already in store") from e
            else:
                return False
    return True


def retrieve_openml(cache_dir: str = None) -> Generator[SupervisedDataset, None, None]:
    """Retrives OpenML CC18 datasets.

    Args:
        cache_dir (str, optional): Use this cache directory across calls. Defaults to None.

    Yields:
        Generator[SupervisedDataset]: Yields datasets.
    """
    import openml

    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir, "openml")
    
    dataset_names_filename = "dataset_names.json"
    dataset_names_stream = retrieve_cache(cache_dir, [dataset_names_filename])
    if dataset_names_stream is None:
        dataset_names = []
        suite = openml.study.get_suite(99)
        tasks = suite.tasks.copy()
        random.Random(1337).shuffle(tasks)
        for task_id in tqdm(tasks, desc="openml"):
            task = openml.tasks.get_task(task_id, download_splits=False, download_data=False, download_qualities=False, download_features_meta_data=False)
            dataset = openml.datasets.get_dataset(task.dataset_id, download_data=True, download_qualities=True, download_features_meta_data=True)
            name = dataset.name
            dataset_names.append(name)

            X_name = f"{name}.X.parquet"
            y_name = f"{name}.y.parquet"
            meta_name = f"{name}.meta.json"
            problem = (
                "binary" if dataset.qualities["NumberOfClasses"] == 2 else "multiclass"
            )
            X, y, categorical_mask, feature_names = dataset.get_data(
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
            if cache_dir is not None:
                serialized = SupervisedDataset.serialize(supervised)
                update_cache(cache_dir, [X_name, y_name, meta_name], serialized)
            yield supervised

        if cache_dir is not None:
            _, dataset_names_stream = BytesParser.serialize({"dataset_names": dataset_names})
            update_cache(cache_dir, [dataset_names_filename], [dataset_names_stream])
    else:
        dataset_names_stream = dataset_names_stream[0]
        dataset_names = BytesParser.deserialize(MIMETYPE_JSON, dataset_names_stream)["dataset_names"]
        for name in tqdm(dataset_names, desc="openml"):
            X_name = f"{name}.X.parquet"
            y_name = f"{name}.y.parquet"
            meta_name = f"{name}.meta.json"
            cached = retrieve_cache(cache_dir, [X_name, y_name, meta_name])
            supervised = SupervisedDataset.deserialize(*cached)
            yield supervised


def retrieve_catboost_50k(cache_dir: str = None) -> Generator[SupervisedDataset, None, None]:
    """Retrieves catboost regression and classification datasets that have less than 50k training instances.

    Does not download adult dataset as currently there some download issues.

    Args:
        cache_dir (str, optional): Use this cache directory across calls. Defaults to None.

    Yields:
        Generator[SupervisedDataset]: Yields datasets.
    """
    from catboost.datasets import amazon, msrank_10k, titanic

    datasets = [
        # NOTE: CatBoost uses an internal dev link for first attempt, slowing the system to a halt - ignoring adult dataset.
        # {
        #     "name": "adult",
        #     "data_fn": adult,
        #     "problem": "classification",
        #     "target": "income",
        # },
        {
            "name": "amazon",
            "data_fn": amazon,
            "problem": "classification",
            "target": "ACTION"
        },
        {
            "name": "msrank_10k",
            "data_fn": msrank_10k,
            "problem": "regression",
            "target": 0
        },
        {
            "name": "titanic",
            "data_fn": titanic,
            "problem": "classification",
            "target": "Survived"
        },
    ]

    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir, "catboost_50k")

    for dataset in tqdm(datasets, desc="catboost_50k"):
        name = dataset['name']
        X_name = f"{name}.X.parquet"
        y_name = f"{name}.y.parquet"
        meta_name = f"{name}.meta.json"

        cached = retrieve_cache(cache_dir, [X_name, y_name, meta_name])
        if cached is None:
            df = dataset['data_fn']()[0]
            target = dataset['target']
            X = df.drop(target, axis=1)
            y = df[target]
            problem = dataset['problem']
            if dataset['problem'] == "classification":
                problem = "binary" if len(y.unique()) == 2 else "multiclass"
            meta = {
                "name": name,
                "problem": problem,
                "source": "catboost_50k",
                "categorical_mask": [dt.kind == "O" for dt in X.dtypes],
                "feature_names": list(X.columns),
            }
            supervised = SupervisedDataset(X, y, meta)
            if cache_dir is not None:
                serialized = SupervisedDataset.serialize(supervised)
                update_cache(cache_dir, [X_name, y_name, meta_name], serialized)
        else:
            supervised = SupervisedDataset.deserialize(*cached)
        yield supervised


def retrieve_pmlb(cache_dir: str = None) -> Generator[SupervisedDataset, None, None]:
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
            if cache_dir is not None:
                serialized = SupervisedDataset.serialize(supervised)
                update_cache(cache_dir, [X_name, y_name, meta_name], serialized)
        else:
            supervised = SupervisedDataset.deserialize(*cached)
        yield supervised
