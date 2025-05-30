# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

import sys

from interpret.ext.extension import GLASSBOX_EXTENSION_KEY, _is_valid_glassbox_explainer
from interpret.ext.extension_utils import load_class_extensions

load_class_extensions(
    sys.modules[__name__], GLASSBOX_EXTENSION_KEY, _is_valid_glassbox_explainer
)
