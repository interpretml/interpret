""" Entity models for benchmarking.
"""

import enum
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
    JSON,
    DateTime,
    Enum,
    Table,
    UniqueConstraint,
)
from sqlalchemy.sql.expression import null
from sqlalchemy.sql.sqltypes import Boolean, Numeric, LargeBinary

# The following are often used as titles, keep length small.
PROBLEM_LEN = 64
NAME_LEN = 256
VERSION_LEN = 256

# Diagnostic fields, length can be long.
ERROR_LEN = 10000

# Measure related fields.
MEASURE_STR_LEN = None

# These descriptions shouldn't be much longer than a tweet (use links for more verbosity).
DESCRIPTION_LEN = 300

# Maximum limit URI length set based on the below link:
# https://stackoverflow.com/questions/417142/what-is-the-maximum-length-of-a-url-in-different-browsers
URI_LEN = 2000

# Media type length based on below link:
# https://datatracker.ietf.org/doc/html/rfc4288#section-4.2
MIMETYPE_LEN = 127

Base = declarative_base()


class TypeEnum(enum.Enum):
    NUMBER = 0
    STR = 1
    JSON = 2


class StatusEnum(enum.Enum):
    READY = 0
    RUNNING = 1
    COMPLETE = 2
    ERROR = 3
    SUSPENDED = 4


task_asset_table = Table(
    "task_asset",
    Base.metadata,
    Column("task_id", ForeignKey("task.id"), primary_key=True),
    Column("asset_id", ForeignKey("asset.id"), primary_key=True),
)
trial_input_asset_table = Table(
    "trial_input_asset",
    Base.metadata,
    Column("trial_id", ForeignKey("trial.id"), primary_key=True),
    Column("asset_id", ForeignKey("asset.id"), primary_key=True),
)
trial_output_asset_table = Table(
    "trial_output_asset",
    Base.metadata,
    Column("trial_id", ForeignKey("trial.id"), primary_key=True),
    Column("asset_id", ForeignKey("asset.id"), primary_key=True),
)
trial_measure_outcome_table = Table(
    "trial_measure_outcome",
    Base.metadata,
    Column("trial_id", ForeignKey("trial.id"), primary_key=True),
    Column("measure_outcome_id", ForeignKey("measure_outcome.id"), primary_key=True),
)
task_measure_outcome_table = Table(
    "task_measure_outcome",
    Base.metadata,
    Column("task_id", ForeignKey("task.id"), primary_key=True),
    Column("measure_outcome_id", ForeignKey("measure_outcome.id"), primary_key=True),
)


class Experiment(Base):
    """The overall experiment, includes access to trials."""

    __tablename__ = "experiment"
    id = Column(Integer, primary_key=True)
    name = Column(String(NAME_LEN), unique=True)
    description = Column(String(DESCRIPTION_LEN))

    trials = relationship("Trial", back_populates="experiment")


class Trial(Base):
    """A single trial replicate, consists primarily of task, method and its results."""

    __tablename__ = "trial"
    id = Column(Integer, primary_key=True)

    experiment_id = Column(Integer, ForeignKey("experiment.id"))
    experiment = relationship("Experiment", back_populates="trials")
    task_id = Column(Integer, ForeignKey("task.id"))
    task = relationship("Task", back_populates="trials")
    method_id = Column(Integer, ForeignKey("method.id"))
    method = relationship("Method", back_populates="trials")
    replicate_num = Column(Integer)
    meta = Column(JSON)

    status = Column(Enum(StatusEnum))
    errmsg = Column(String(ERROR_LEN), nullable=True)
    create_time = Column(DateTime)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)

    measure_outcomes = relationship(
        "MeasureOutcome", secondary=trial_measure_outcome_table, back_populates="trials"
    )
    input_assets = relationship(
        "Asset", secondary=trial_input_asset_table, back_populates="trial_inputs"
    )
    output_assets = relationship(
        "Asset", secondary=trial_output_asset_table, back_populates="trial_outputs"
    )


class MeasureDescription(Base):
    """Describes a measure that could be generated in a trial."""

    __tablename__ = "measure_description"
    id = Column(Integer, primary_key=True)
    name = Column(String(NAME_LEN), unique=True)
    description = Column(String(DESCRIPTION_LEN))
    type = Column(Enum(TypeEnum))
    lower_is_better = Column(Boolean)


class MeasureOutcome(Base):
    """The recording of a measure generated in a trial."""

    __tablename__ = "measure_outcome"
    id = Column(Integer, primary_key=True)

    measure_description_id = Column(Integer, ForeignKey("measure_description.id"))
    measure_description = relationship("MeasureDescription")

    timestamp = Column(DateTime)
    seq_num = Column(Integer)
    num_val = Column(Numeric, nullable=True)
    str_val = Column(String(MEASURE_STR_LEN), nullable=True)
    json_val = Column(JSON, nullable=True)

    trials = relationship(
        "Trial",
        secondary=trial_measure_outcome_table,
        back_populates="measure_outcomes",
    )
    tasks = relationship(
        "Task", secondary=task_measure_outcome_table, back_populates="measure_outcomes"
    )


class Method(Base):
    """A method/technique/treatment that is being studied in a trial."""

    __tablename__ = "method"
    id = Column(Integer, primary_key=True)
    name = Column(String(NAME_LEN), unique=True)
    description = Column(String(DESCRIPTION_LEN))
    version = Column(String(VERSION_LEN))
    params = Column(JSON)
    env = Column(JSON)
    trials = relationship("Trial", back_populates="method")


class Task(Base):
    """A problem tied with a dataset. I.e. regression on Boston data."""

    __tablename__ = "task"
    id = Column(Integer, primary_key=True)
    name = Column(String(NAME_LEN))
    description = Column(String(DESCRIPTION_LEN))
    version = Column(String(VERSION_LEN))
    problem = Column(String(PROBLEM_LEN))
    origin = Column(String(NAME_LEN))
    config = Column(JSON)

    trials = relationship("Trial", back_populates="task")
    assets = relationship("Asset", secondary=task_asset_table, back_populates="tasks")
    measure_outcomes = relationship(
        "MeasureOutcome", secondary=task_measure_outcome_table, back_populates="tasks"
    )
    __table_args__ = (UniqueConstraint("name", "problem", "origin", name="u_name_problem_origin"),)


class Asset(Base):
    """An asset with its associated mimetype and uri."""

    __tablename__ = "asset"
    id = Column(Integer, primary_key=True)

    name = Column(String(NAME_LEN))
    description = Column(String(DESCRIPTION_LEN))
    version = Column(String(VERSION_LEN))

    is_embedded = Column(Boolean)
    embedded = Column(LargeBinary, nullable=True)
    uri = Column(String(URI_LEN), nullable=True)
    mimetype = Column(String(MIMETYPE_LEN))

    trial_inputs = relationship(
        "Trial", secondary=trial_input_asset_table, back_populates="input_assets"
    )
    trial_outputs = relationship(
        "Trial", secondary=trial_output_asset_table, back_populates="output_assets"
    )
    tasks = relationship("Task", secondary=task_asset_table, back_populates="assets")
