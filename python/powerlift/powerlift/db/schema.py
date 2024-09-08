""" Entity models for benchmarking.
"""

import enum
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Text,
    Index,
    ForeignKey,
    DateTime,
    Enum,
    Table,
    UniqueConstraint,
    text,
)
from sqlalchemy.sql.expression import null
from sqlalchemy.sql.sqltypes import Boolean, Numeric, LargeBinary

# The following are often used as titles, keep length small.
PROBLEM_LEN = 256
NAME_LEN = 256

# These descriptions shouldn't be much longer than a tweet (use links for more verbosity).
DESCRIPTION_LEN = 1024

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


class Experiment(Base):
    """The overall experiment, includes access to trials."""

    __tablename__ = "experiment"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    name = Column(String(NAME_LEN), unique=True, nullable=False)
    description = Column(String(DESCRIPTION_LEN), nullable=True)
    shell_install = Column(Text, nullable=True)
    pip_install = Column(Text, nullable=True)
    script = Column(Text, nullable=False)
    trial_fn = Column(Text, nullable=False)

    __table_args__ = (Index("ix_name", "name"),)

    wheels = relationship("Wheel", back_populates="experiment")
    trials = relationship("Trial", back_populates="experiment")


class Trial(Base):
    """A single trial replicate, consists primarily of task, method and its results."""

    __tablename__ = "trial"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)

    experiment_id = Column(Integer, ForeignKey("experiment.id"), nullable=False)
    experiment = relationship("Experiment", back_populates="trials")

    runner_id = Column(Integer, nullable=True)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)

    create_time = Column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    errmsg = Column(Text, nullable=True)

    task_id = Column(Integer, ForeignKey("task.id"), nullable=False)
    task = relationship("Task", back_populates="trials")

    method = Column(String(NAME_LEN), nullable=False)
    meta = Column(Text, nullable=False)
    replicate_num = Column(Integer, nullable=False)

    measure_outcomes = relationship("MeasureOutcome", back_populates="trials")
    __table_args__ = (
        # We select work by exact runner_id match, or when both runner_id and
        # start_time are NULL. Both ways are fast to find with this index.
        # experiment_id is included in the index for the WHERE clause in queries.
        Index(
            "ix_order",
            experiment_id,
            runner_id,
            start_time,
            id,  # include id to order them when runner_id and start_time are NULL
        ),
    )


class MeasureOutcome(Base):
    """The recording of a measure generated in a trial."""

    __tablename__ = "measure_outcome"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)

    # Do not include an index for trial_id to speed insertion from multiple runners.
    # We would only use the index when querying in the get_status but that can use
    # a table scan since normally it will return a significant percentage of the
    # rows anyways.
    trial_id = Column(Integer, ForeignKey("trial.id"), nullable=False)

    timestamp = Column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )

    name = Column(String(NAME_LEN), nullable=False)
    type = Column(Integer, nullable=False)
    seq_num = Column(Integer, nullable=False)
    val = Column(Text, nullable=False)

    trials = relationship(
        "Trial",
        back_populates="measure_outcomes",
    )


class Task(Base):
    """A problem tied with a dataset. I.e. regression on Boston data."""

    __tablename__ = "task"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    name = Column(String(NAME_LEN), nullable=False)
    problem = Column(String(PROBLEM_LEN), nullable=False)
    origin = Column(String(NAME_LEN), nullable=False)

    n_samples = Column(Integer, nullable=False)
    n_features = Column(Integer, nullable=False)
    n_classes = Column(Integer, nullable=False)

    max_unique_continuous = Column(Integer, nullable=False)
    max_categories = Column(Integer, nullable=False)
    total_categories = Column(Integer, nullable=False)
    percent_categorical = Column(Float, nullable=False)
    percent_special_values = Column(Float, nullable=False)

    meta = Column(Text, nullable=False)
    x = Column(LargeBinary, nullable=False)
    y = Column(LargeBinary, nullable=False)

    trials = relationship("Trial", back_populates="task")
    __table_args__ = (
        UniqueConstraint("name", "problem", "origin", name="u_name_problem_origin"),
    )


class Wheel(Base):
    """Wheel assets for experiments."""

    __tablename__ = "wheel"
    experiment_id = Column(
        Integer, ForeignKey("experiment.id"), primary_key=True, nullable=False
    )
    name = Column(String(NAME_LEN), primary_key=True, nullable=False)
    embedded = Column(LargeBinary, nullable=False)
    experiment = relationship("Experiment", back_populates="wheels")
