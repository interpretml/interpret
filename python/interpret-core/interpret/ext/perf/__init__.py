# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import sys

from ..extension import PERF_EXTENSION_KEY, _is_valid_perf_explainer
from ..extension_utils import load_class_extensions

load_class_extensions(
    sys.modules[__name__], PERF_EXTENSION_KEY, _is_valid_perf_explainer
)
