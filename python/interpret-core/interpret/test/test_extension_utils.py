# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import pytest
from interpret.ext.extension_utils import _validate_class_name


@pytest.mark.parametrize(
    "name_and_is_valid",
    [
        ("aagdsg.afds", False),
        ("1332", False),
        ("&sgag", False),
        ("aaaaaa", True),
        (",,,,", False),
        ("", False),
        ("_AmValid", True),
        ("_AM_NOT_NICE_BUT_VALID", True),
    ],
)
def test_name_validation(name_and_is_valid):
    name, is_valid = name_and_is_valid
    if is_valid:
        _validate_class_name(name)
    else:
        with pytest.raises(ValueError):
            _validate_class_name(name)
