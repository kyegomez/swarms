import pytest
from unittest.mock import patch
from swarms.memory.pg import PgVectorVectorStore
from dotenv import load_dotenv
import os

load_dotenv()


PSG_CONNECTION_STRING = os.getenv("PSG_CONNECTION_STRING")


def test_init():
    with patch("sqlalchemy.create_engine") as MockEngine:
        store = PgVectorVectorStore(
            connection_string=PSG_CONNECTION_STRING,
            table_name="test",
        )
        MockEngine.assert_called_once()
        assert store.engine == MockEngine.return_value


def test_init_exception():
    with pytest.raises(ValueError):
        PgVectorVectorStore(
            connection_string=(
                "mysql://root:password@localhost:3306/test"
            ),
            table_name="test",
        )


def test_setup():
    with patch("sqlalchemy.create_engine") as MockEngine:
        store = PgVectorVectorStore(
            connection_string=PSG_CONNECTION_STRING,
            table_name="test",
        )
        store.setup()
        MockEngine.execute.assert_called()


def test_upsert_vector():
    with patch("sqlalchemy.create_engine"), patch(
        "sqlalchemy.orm.Session"
    ) as MockSession:
        store = PgVectorVectorStore(
            connection_string=PSG_CONNECTION_STRING,
            table_name="test",
        )
        store.upsert_vector(
            [1.0, 2.0, 3.0],
            "test_id",
            "test_namespace",
            {"meta": "data"},
        )
        MockSession.assert_called()
        MockSession.return_value.merge.assert_called()
        MockSession.return_value.commit.assert_called()


def test_load_entry():
    with patch("sqlalchemy.create_engine"), patch(
        "sqlalchemy.orm.Session"
    ) as MockSession:
        store = PgVectorVectorStore(
            connection_string=PSG_CONNECTION_STRING,
            table_name="test",
        )
        store.load_entry("test_id", "test_namespace")
        MockSession.assert_called()
        MockSession.return_value.get.assert_called()


def test_load_entries():
    with patch("sqlalchemy.create_engine"), patch(
        "sqlalchemy.orm.Session"
    ) as MockSession:
        store = PgVectorVectorStore(
            connection_string=PSG_CONNECTION_STRING,
            table_name="test",
        )
        store.load_entries("test_namespace")
        MockSession.assert_called()
        MockSession.return_value.query.assert_called()
        MockSession.return_value.query.return_value.filter_by.assert_called()
        MockSession.return_value.query.return_value.all.assert_called()


def test_query():
    with patch("sqlalchemy.create_engine"), patch(
        "sqlalchemy.orm.Session"
    ) as MockSession:
        store = PgVectorVectorStore(
            connection_string=PSG_CONNECTION_STRING,
            table_name="test",
        )
        store.query("test_query", 10, "test_namespace")
        MockSession.assert_called()
        MockSession.return_value.query.assert_called()
        MockSession.return_value.query.return_value.filter_by.assert_called()
        MockSession.return_value.query.return_value.limit.assert_called()
        MockSession.return_value.query.return_value.all.assert_called()
