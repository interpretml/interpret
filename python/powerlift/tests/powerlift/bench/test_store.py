from powerlift.bench.store import (
    Store,
    retrieve_openml,
    retrieve_pmlb,
    populate_with_datasets,
)
import pandas as pd
from powerlift import db
import pytest
from itertools import islice


@pytest.mark.skip("high bandwidth test")
def test_retrieve_openml(cache_dir, dataset_limit):
    # Do it twice to trigger cache
    for _ in range(2):
        count = 0
        for supervised in islice(retrieve_openml(cache_dir), dataset_limit):
            X, y, meta = supervised.X, supervised.y, supervised.meta
            assert meta["name"] != ""
            assert meta["problem"] in ["binary", "multiclass"]
            assert len(meta["feature_names"]) > 0
            assert len(meta["categorical_mask"]) > 0
            assert meta["source"] == "openml"
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)
            count += 1
        assert count > 0


@pytest.mark.skip("high bandwidth test")
def test_retrieve_pmlb(cache_dir, dataset_limit):
    # Do it twice to trigger cache
    for _ in range(2):
        count = 0
        for supervised in islice(retrieve_pmlb(cache_dir), dataset_limit):
            X, y, meta = supervised.X, supervised.y, supervised.meta
            assert meta["name"] != ""
            assert meta["problem"] in ["binary", "multiclass", "regression"]
            assert len(meta["feature_names"]) > 0
            assert len(meta["categorical_mask"]) > 0
            assert meta["source"] == "pmlb"
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)
            count += 1
        assert count > 0


@pytest.mark.skip("high bandwidth test")
def test_populate_with_datasets(store: Store, dataset_limit):
    from sqlalchemy import func

    dataset_iter = islice(retrieve_openml(), dataset_limit)
    populate_with_datasets(store, dataset_iter)

    assert store._session.query(func.count(db.Asset.id)).scalar()
    assert store._session.query(func.count(db.Task.id)).scalar()
