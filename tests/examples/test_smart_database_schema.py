import json
import runpy
import sqlite3
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/demos/apps/smart_database_swarm.py"
)


@pytest.mark.parametrize(
    "table_names", [[], ["employees"], ["order_items", "x"]]
)
def test_schema_lists_public_tables(
    table_names: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """List actual tables while hiding underscore-prefixed metadata."""
    swarms = ModuleType("swarms")
    swarms.Agent = Mock()
    swarms.HierarchicalSwarm = Mock()
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = Mock()
    loguru = ModuleType("loguru")
    loguru.logger = Mock()
    for module in (swarms, dotenv, loguru):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    example = runpy.run_path(str(EXAMPLE))

    database = tmp_path / "company.db"
    connection = sqlite3.connect(database)
    try:
        with connection:
            connection.execute(
                "CREATE TABLE _database_metadata (key TEXT)"
            )
            for name in table_names:
                connection.execute(
                    f"CREATE TABLE {name} (id INTEGER PRIMARY KEY, "
                    "name TEXT NOT NULL)"
                )
                connection.execute(
                    f"INSERT INTO {name} VALUES (?, ?)", (1, "Ada")
                )
    finally:
        connection.close()

    result = json.loads(example["get_database_schema"](str(database)))

    assert result["status"] == "success"
    schema = result["schema"]
    assert schema["table_count"] == len(table_names)
    assert set(schema["tables"]) == set(table_names)
    for table in schema["tables"].values():
        assert table["row_count"] == 1
        assert table["column_count"] == 2
        assert [column["name"] for column in table["columns"]] == [
            "id",
            "name",
        ]
