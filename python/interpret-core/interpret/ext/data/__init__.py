# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import sys

from interpret.ext.extension import DATA_EXTENSION_KEY, _is_valid_data_explainer
from interpret.ext.extension_utils import load_class_extensions

load_class_extensions(
    sys.modules[__name__], DATA_EXTENSION_KEY, _is_valid_data_explainer
)
