import csv
import json
import runpy
from pathlib import Path

import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/demos/synthetic_data/profession_sim"
    / "convert_json_to_csv.py"
)


@pytest.mark.parametrize(
    "professions",
    [
        [
            {"profession": "Engineer", "salary": 100},
            {"salary": 200, "profession": "Doctor"},
        ],
        [
            {"profession": "Engineer", "salary": 100},
            {"profession": 'Writer, "editor"', "salary": 200},
        ],
        [],
    ],
    ids=["reordered-keys", "matching-keys", "empty-input"],
)
def test_profession_csv_columns(
    professions: list[dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep each profession value under its named CSV column."""
    source = tmp_path / "profession_personas.progress.json"
    source.write_text(json.dumps({"professions": professions}))
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(EXAMPLE))

    with (tmp_path / "data_personas_progress.csv").open(
        newline=""
    ) as output:
        reader = csv.DictReader(output)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames == (
        list(professions[0]) if professions else None
    )
    assert rows == [
        {key: str(value) for key, value in row.items()}
        for row in professions
    ]
