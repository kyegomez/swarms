from pathlib import Path
from zipfile import ZipFile

import pytest

from swarms.utils.file_processing import zip_folders


@pytest.mark.parametrize(
    ("second_name", "second_archive_name"),
    [("notes", "notes"), ("reports", "reports_2")],
)
def test_zip_folders_preserves_both_inputs(
    tmp_path: Path, second_name: str, second_archive_name: str
) -> None:
    """Store each folder separately, even when both have the same name."""
    first = tmp_path / "first" / "reports"
    second = tmp_path / "second" / second_name
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "summary.txt").write_text("first", encoding="utf-8")
    (second / "summary.txt").write_text("second", encoding="utf-8")
    output = tmp_path / "combined"

    zip_folders(str(first), str(second), str(output))

    with ZipFile(output.with_suffix(".zip")) as archive:
        assert archive.read("reports/summary.txt") == b"first"
        assert (
            archive.read(f"{second_archive_name}/summary.txt")
            == b"second"
        )
