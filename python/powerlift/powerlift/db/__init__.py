"""Database schema for benchmarks.

Currently we use SQLAlchemy. The basics of it are well explained in the below link:
https://docs.sqlalchemy.org/en/14/tutorial/index.html

Design-wise we balance between design simplicity, performance and wide environment support.
Ideally, we can run the same benchmarks on both a single machine and distributed systems
with minimal installations (only Python dependencies at the moment).

The database component is currently tested on both sqlite and postgresql.

TODO:
- Consider migration
"""

from .actions import create_db, delete_db
from .schema import (
    DESCRIPTION_LEN,
    NAME_LEN,
    PROBLEM_LEN,
    Experiment,
    MeasureOutcome,
    StatusEnum,
    Task,
    Trial,
    TypeEnum,
)
