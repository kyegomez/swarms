import json
import os
import tempfile
from datetime import datetime
from unittest.mock import mock_open, patch

import pytest

from swarms.artifacts.main_artifact import Artifact, FileVersion


# ============================================================================
# Core model tests
# ============================================================================


def test_file_version():
    version = FileVersion(
        version_number=1,
        content="Initial content",
        timestamp=datetime.now(),
    )
    assert version.version_number == 1
    assert version.content == "Initial content"


def test_artifact_creation():
    artifact = Artifact(file_path="test.txt", file_type=".txt")
    assert artifact.file_path == "test.txt"
    assert artifact.file_type == ".txt"
    assert artifact.contents == ""
    assert artifact.versions == []
    assert artifact.edit_count == 0


def test_artifact_create():
    artifact = Artifact(file_path="test.txt", file_type=".txt")
    artifact.create("Initial content")
    assert artifact.contents == "Initial content"
    assert len(artifact.versions) == 1
    assert artifact.versions[0].content == "Initial content"
    assert artifact.edit_count == 0


def test_artifact_edit():
    artifact = Artifact(file_path="test.txt", file_type=".txt")
    artifact.create("Initial content")
    artifact.edit("First edit")
    assert artifact.contents == "First edit"
    assert len(artifact.versions) == 2
    assert artifact.versions[1].content == "First edit"
    assert artifact.edit_count == 1


def test_artifact_get_version():
    artifact = Artifact(file_path="test.txt", file_type=".txt")
    artifact.create("Initial content")
    artifact.edit("First edit")
    version = artifact.get_version(1)
    assert version.content == "Initial content"


def test_artifact_get_contents():
    artifact = Artifact(file_path="test.txt", file_type=".txt")
    artifact.create("Initial content")
    assert artifact.get_contents() == "Initial content"


def test_artifact_get_version_history():
    artifact = Artifact(file_path="test.txt", file_type=".txt")
    artifact.create("Initial content")
    artifact.edit("First edit")
    history = artifact.get_version_history()
    assert "Version 1" in history
    assert "Version 2" in history


def test_artifact_to_dict():
    artifact = Artifact(file_path="test.txt", file_type=".txt")
    artifact.create("Initial content")
    artifact_dict = artifact.to_dict()
    assert artifact_dict["file_path"] == "test.txt"
    assert artifact_dict["file_type"] == ".txt"
    assert artifact_dict["contents"] == "Initial content"
    assert artifact_dict["edit_count"] == 0


def test_artifact_from_dict():
    artifact_dict = {
        "file_path": "test.txt",
        "file_type": ".txt",
        "contents": "Initial content",
        "versions": [
            {
                "version_number": 1,
                "content": "Initial content",
                "timestamp": datetime.now().isoformat(),
            }
        ],
        "edit_count": 0,
    }
    artifact = Artifact.from_dict(artifact_dict)
    assert artifact.file_path == "test.txt"
    assert artifact.file_type == ".txt"
    assert artifact.contents == "Initial content"
    assert artifact.versions[0].content == "Initial content"
    assert artifact.edit_count == 0


# ============================================================================
# save_as / export tests (filesystem)
# ============================================================================


@pytest.fixture
def saved_artifact(tmp_path):
    """Provide an Artifact backed by a real temp file with multi-line content."""
    file_path = str(tmp_path / "test_file.txt")
    content = "This is test content\nWith multiple lines"
    artifact = Artifact(
        file_path=file_path,
        file_type=".txt",
        contents=content,
        edit_count=0,
    )
    artifact.create(content)
    return artifact, file_path, content, str(tmp_path)


def test_save_as_txt(saved_artifact):
    """Saving artifact as .txt writes the raw content."""
    artifact, file_path, content, _ = saved_artifact
    output_path = os.path.splitext(file_path)[0] + ".txt"

    artifact.save_as(".txt")

    assert os.path.exists(output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        assert f.read() == content


def test_save_as_markdown(saved_artifact):
    """Saving as .md wraps content with a markdown header."""
    artifact, file_path, content, _ = saved_artifact
    output_path = os.path.splitext(file_path)[0] + ".md"

    artifact.save_as(".md")

    assert os.path.exists(output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        out = f.read()
    assert content in out
    assert "# test_file.txt" in out


def test_save_as_python(saved_artifact):
    """Saving as .py wraps content in a docstring with a generation note."""
    artifact, file_path, content, _ = saved_artifact
    output_path = os.path.splitext(file_path)[0] + ".py"

    artifact.save_as(".py")

    assert os.path.exists(output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        out = f.read()
    assert content in out
    assert '"""' in out
    assert "Generated Python file" in out


def test_file_writing_called(saved_artifact):
    """save_as triggers an actual file write with the artifact's contents."""
    artifact, _, content, _ = saved_artifact
    with patch("builtins.open", new_callable=mock_open) as mock_file:
        artifact.save_as(".txt")
        mock_file.assert_called()
        mock_file().write.assert_called_with(content)


def test_invalid_format(saved_artifact):
    """Unknown file extension must raise ValueError."""
    artifact, _, _, _ = saved_artifact
    with pytest.raises(ValueError):
        artifact.save_as(".invalid")


def test_export_import_json(saved_artifact):
    """Round-trip via JSON export + Artifact(**data) preserves content."""
    artifact, _, content, tmp_dir = saved_artifact
    json_path = os.path.join(tmp_dir, "test.json")

    artifact.export_to_json(json_path)
    assert os.path.exists(json_path)

    with open(json_path, "r") as f:
        data = json.loads(f.read())

    imported = Artifact(**data)
    assert imported.contents == content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
