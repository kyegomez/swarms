import os
from unittest.mock import patch

from dotenv import load_dotenv

from swarms.memory.pg import PostgresDB

load_dotenv()

PSG_CONNECTION_STRING = os.getenv("PSG_CONNECTION_STRING")


def test_init():
    with patch("sqlalchemy.create_engine") as MockEngine:
        db = PostgresDB(
            connection_string=PSG_CONNECTION_STRING,
            table_name="test",
        )
        MockEngine.assert_called_once()
        assert db.engine == MockEngine.return_value


def test_create_vector_model():
    with patch("sqlalchemy.create_engine"):
        db = PostgresDB(
            connection_string=PSG_CONNECTION_STRING,
            table_name="test",
        )
        model = db._create_vector_model()
        assert model.__tablename__ == "test"


def test_add_or_update_vector():
    with patch("sqlalchemy.create_engine"), patch(
        "sqlalchemy.orm.Session"
    ) as MockSession:
        db = PostgresDB(
            connection_string=PSG_CONNECTION_STRING,
            table_name="test",
        )
        db.add_or_update_vector(
            "test_vector",
            "test_id",
            "test_namespace",
            {"meta": "data"},
        )
        MockSession.assert_called()
        MockSession.return_value.merge.assert_called()
        MockSession.return_value.commit.assert_called()


def test_query_vectors():
    with patch("sqlalchemy.create_engine"), patch(
        "sqlalchemy.orm.Session"
    ) as MockSession:
        db = PostgresDB(
            connection_string=PSG_CONNECTION_STRING,
            table_name="test",
        )
        db.query_vectors("test_query", "test_namespace")
        MockSession.assert_called()
        MockSession.return_value.query.assert_called()
        MockSession.return_value.query.return_value.filter_by.assert_called()
        MockSession.return_value.query.return_value.filter.assert_called()
        MockSession.return_value.query.return_value.all.assert_called()


def test_delete_vector():
    with patch("sqlalchemy.create_engine"), patch(
        "sqlalchemy.orm.Session"
    ) as MockSession:
        db = PostgresDB(
            connection_string=PSG_CONNECTION_STRING,
            table_name="test",
        )
        db.delete_vector("test_id")
        MockSession.assert_called()
        MockSession.return_value.get.assert_called()
        MockSession.return_value.delete.assert_called()
        MockSession.return_value.commit.assert_called()
