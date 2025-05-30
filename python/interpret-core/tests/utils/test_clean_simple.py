# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import numpy as np
from interpret.utils._clean_simple import clean_dimensions


def test_clean_dimensions_2d():
    init_score = [
        [[[((x,) for x in [1, 2])]]],
        [3, 4],
        (np.array([5, 6]),),
        np.array([[[(7,), (8,)]]]),
    ]
    init_score = clean_dimensions(init_score, "init_score")
    init_score = init_score.astype(np.float64, copy=False)
    assert init_score.shape == (4, 2)
    assert init_score[0, 0] == 1
    assert init_score[0, 1] == 2
    assert init_score[1, 0] == 3
    assert init_score[1, 1] == 4
    assert init_score[2, 0] == 5
    assert init_score[2, 1] == 6
    assert init_score[3, 0] == 7
    assert init_score[3, 1] == 8
