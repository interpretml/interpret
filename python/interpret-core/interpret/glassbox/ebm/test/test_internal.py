# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..internal import Native, NativeEBMBooster

import numpy as np
import ctypes as ct
from contextlib import closing

def test_booster_internals():
    with closing(
        NativeEBMBooster(
            model_type="classification",
            n_classes=2,
            features_categorical=np.array([0], dtype=ct.c_int64, order="C"), 
            features_bin_count=np.array([2], dtype=ct.c_int64, order="C"),
            feature_groups=[[0]],
            X_train=np.array([[0]], dtype=ct.c_int64, order="C"),
            y_train=np.array([0], dtype=ct.c_int64, order="C"),
            scores_train=None,
            X_val=np.array([[0]], dtype=ct.c_int64, order="C"),
            y_val=np.array([0], dtype=ct.c_int64, order="C"),
            scores_val=None,
            n_inner_bags=0,
            random_state=42,
            optional_temp_params=None,
        )
    ) as native_ebm_booster:
        gain = native_ebm_booster.generate_model_update(
            feature_group_index=0,
            generate_update_options=Native.GenerateUpdateOptions_Default,
            learning_rate=0.01,
            min_samples_leaf=2,
            max_leaves=np.array([2], dtype=ct.c_int64, order="C"),
        )
        assert gain == 0

        cuts = native_ebm_booster.get_model_update_cuts()
        assert len(cuts) == 1
        assert len(cuts[0]) == 0

        model_update = native_ebm_booster.get_model_update_expanded()
        assert len(model_update.shape) == 1
        assert model_update.shape[0] == 2
        assert model_update[0] < 0

        native_ebm_booster.set_model_update_expanded(0, model_update)

        metric = native_ebm_booster.apply_model_update()
        assert 0 < metric

        model = native_ebm_booster.get_best_model()
        assert len(model) == 1
        assert len(model[0].shape) == 1
        assert model[0].shape[0] == 2
        assert model[0][0] < 0


def test_one_class():
    with closing(
        NativeEBMBooster(
            model_type="classification",
            n_classes=1,
            features_categorical=np.array([0], dtype=ct.c_int64, order="C"), 
            features_bin_count=np.array([2], dtype=ct.c_int64, order="C"),
            feature_groups=[[0]],
            X_train=np.array([[0, 1, 0]], dtype=ct.c_int64, order="C"),
            y_train=np.array([0, 0, 0], dtype=ct.c_int64, order="C"),
            scores_train=None,
            X_val=np.array([[1, 0, 1]], dtype=ct.c_int64, order="C"),
            y_val=np.array([0, 0, 0], dtype=ct.c_int64, order="C"),
            scores_val=None,
            n_inner_bags=0,
            random_state=42,
            optional_temp_params=None,
        )
    ) as native_ebm_booster:
        gain = native_ebm_booster.generate_model_update(
            feature_group_index=0,
            generate_update_options=Native.GenerateUpdateOptions_Default,
            learning_rate=0.01,
            min_samples_leaf=2,
            max_leaves=np.array([2], dtype=ct.c_int64, order="C"),
        )
        assert gain == 0

        cuts = native_ebm_booster.get_model_update_cuts()
        assert len(cuts) == 1
        assert len(cuts[0]) == 0

        model_update = native_ebm_booster.get_model_update_expanded()
        assert model_update is None

        native_ebm_booster.set_model_update_expanded(0, model_update)

        metric = native_ebm_booster.apply_model_update()
        assert metric == 0

        model = native_ebm_booster.get_best_model()
        assert len(model) == 1
        assert model[0] is None
