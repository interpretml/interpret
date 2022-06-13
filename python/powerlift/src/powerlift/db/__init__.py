""" Database schema for benchmarks.

Currently we use SQLAlchemy. The basics of it are well explained in the below link:
https://docs.sqlalchemy.org/en/14/tutorial/index.html

Design-wise we balance between design simplicity, performance and wide environment support.
Ideally, we can run the same benchmarks on both a single machine and distributed systems
with minimal installations (only Python dependencies at the moment).

The database component is currently tested on both sqlite and postgresql.

TODO:
- Consider migration
"""

from .schema import (
    Experiment,
    Trial,
    Task,
    Method,
    MeasureDescription,
    MeasureOutcome,
    Asset,
)
from .schema import NAME_LEN, PROBLEM_LEN, ERROR_LEN, MEASURE_STR_LEN, DESCRIPTION_LEN
from .schema import URI_LEN, MIMETYPE_LEN, TypeEnum, StatusEnum

from .actions import create_db, delete_db
