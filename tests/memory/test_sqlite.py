import sqlite3

import pytest

from swarms.memory.sqlite import SQLiteDB


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)"
    )
    conn.commit()
    return SQLiteDB(":memory:")


def test_add(db):
    db.add("INSERT INTO test (name) VALUES (?)", ("test",))
    result = db.query("SELECT * FROM test")
    assert result == [(1, "test")]


def test_delete(db):
    db.add("INSERT INTO test (name) VALUES (?)", ("test",))
    db.delete("DELETE FROM test WHERE name = ?", ("test",))
    result = db.query("SELECT * FROM test")
    assert result == []


def test_update(db):
    db.add("INSERT INTO test (name) VALUES (?)", ("test",))
    db.update(
        "UPDATE test SET name = ? WHERE name = ?", ("new", "test")
    )
    result = db.query("SELECT * FROM test")
    assert result == [(1, "new")]


def test_query(db):
    db.add("INSERT INTO test (name) VALUES (?)", ("test",))
    result = db.query("SELECT * FROM test WHERE name = ?", ("test",))
    assert result == [(1, "test")]


def test_execute_query(db):
    db.add("INSERT INTO test (name) VALUES (?)", ("test",))
    result = db.execute_query(
        "SELECT * FROM test WHERE name = ?", ("test",)
    )
    assert result == [(1, "test")]


def test_add_without_params(db):
    with pytest.raises(sqlite3.ProgrammingError):
        db.add("INSERT INTO test (name) VALUES (?)")


def test_delete_without_params(db):
    with pytest.raises(sqlite3.ProgrammingError):
        db.delete("DELETE FROM test WHERE name = ?")


def test_update_without_params(db):
    with pytest.raises(sqlite3.ProgrammingError):
        db.update("UPDATE test SET name = ? WHERE name = ?")


def test_query_without_params(db):
    with pytest.raises(sqlite3.ProgrammingError):
        db.query("SELECT * FROM test WHERE name = ?")


def test_execute_query_without_params(db):
    with pytest.raises(sqlite3.ProgrammingError):
        db.execute_query("SELECT * FROM test WHERE name = ?")


def test_add_with_wrong_query(db):
    with pytest.raises(sqlite3.OperationalError):
        db.add("INSERT INTO wrong (name) VALUES (?)", ("test",))


def test_delete_with_wrong_query(db):
    with pytest.raises(sqlite3.OperationalError):
        db.delete("DELETE FROM wrong WHERE name = ?", ("test",))


def test_update_with_wrong_query(db):
    with pytest.raises(sqlite3.OperationalError):
        db.update(
            "UPDATE wrong SET name = ? WHERE name = ?",
            ("new", "test"),
        )


def test_query_with_wrong_query(db):
    with pytest.raises(sqlite3.OperationalError):
        db.query("SELECT * FROM wrong WHERE name = ?", ("test",))


def test_execute_query_with_wrong_query(db):
    with pytest.raises(sqlite3.OperationalError):
        db.execute_query(
            "SELECT * FROM wrong WHERE name = ?", ("test",)
        )
