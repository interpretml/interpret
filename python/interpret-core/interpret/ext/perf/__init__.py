# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import sys
from interpret.ext.extension_utils import load_class_extensions
from interpret.ext.extension import PERF_EXTENSION_KEY, _is_valid_perf_explainer

load_class_extensions(
    sys.modules[__name__], PERF_EXTENSION_KEY, _is_valid_perf_explainer
)
