import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError

import pytest

import swarms


def test_version_attribute_exists():
    assert isinstance(swarms.__version__, str)
    assert swarms.__version__


def test_version_matches_installed_distribution():
    from importlib.metadata import version

    assert swarms.__version__ == version("swarms")


def test_version_is_pep440_shaped():
    assert re.match(
        r"^\d+\.\d+", swarms.__version__
    ), f"expected a numeric version, got {swarms.__version__!r}"


def test_hasattr_reports_true():
    assert hasattr(swarms, "__version__")


def test_version_is_advertised_by_dir_before_access():
    """dir() lists it even in a process that has never read it."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import swarms; print('__version__' in dir(swarms))",
        ],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().endswith("True"), result.stderr


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError) as excinfo:
        swarms.definitely_not_a_real_attribute

    assert "definitely_not_a_real_attribute" in str(excinfo.value)


def test_version_is_cached_after_first_access():
    """The second read comes from globals(), not a fresh metadata scan."""
    swarms.__version__
    assert "__version__" in vars(swarms)


def test_falls_back_when_distribution_is_absent(monkeypatch):
    """A source checkout with no installed dist reports 'unknown'."""
    import importlib.metadata as metadata

    monkeypatch.delitem(vars(swarms), "__version__", raising=False)

    def _raise(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", _raise)

    assert swarms.__getattr__("__version__") == "unknown"

    monkeypatch.undo()
    vars(swarms).pop("__version__", None)


if __name__ == "__main__":
    pytest.main([__file__])
