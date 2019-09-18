# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import sys
from interpret.ext.extension_utils import load_class_extensions
from interpret.ext.extension import PROVIDER_EXTENSION_KEY, _is_valid_provider

load_class_extensions(sys.modules[__name__], PROVIDER_EXTENSION_KEY, _is_valid_provider)
