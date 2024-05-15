""" Actions that can be performed on the database. """

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, drop_database, create_database
from . import schema as db


def delete_db(uri, **create_engine_kwargs):
    """Deletes database if it exists.

    Args:
      uri: Database connection uri.
      **create_engine_kwargs: Keyword args passed to sqlalchemy.create_engine method.
    Returns:
      SQLAlchemy engine.
    """

    engine = create_engine(uri, **create_engine_kwargs)
    if database_exists(engine.url):
        drop_database(engine.url)
    return engine


def create_tables(engine):
    """Creates tables if needed.

    Args:
      engine: SQLAlchemy engine.
    """
    db.Base.metadata.create_all(engine, checkfirst=True)


def drop_tables(engine):
    """Drops tables if needed.

    Args:
      engine: SQLAlchemy engine.
    """
    db.Base.metadata.drop_all(engine, checkfirst=True)


def create_db(uri, **create_engine_kwargs):
    """Creates database if not already exists.

    Args:
      uri: Database connection uri.
      **create_engine_kwargs: Keyword args passed to sqlalchemy.create_engine method.
    Returns:
      SQLAlchemy engine.
    """

    engine = create_engine(uri, **create_engine_kwargs)
    if not database_exists(engine.url):
        create_database(engine.url)
    return engine
