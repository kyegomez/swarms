import re
import subprocess
import sys
from importlib import metadata
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
        check=False,
    )
    assert result.stdout.strip().endswith("True"), result.stderr


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError) as excinfo:
        _ = swarms.definitely_not_a_real_attribute

    assert "definitely_not_a_real_attribute" in str(excinfo.value)


def test_version_is_cached_after_first_access():
    """The second read comes from globals(), not a fresh metadata scan."""
    _ = swarms.__version__
    assert "__version__" in vars(swarms)


def test_falls_back_when_distribution_is_absent(monkeypatch):
    """A source checkout with no installed dist reports 'unknown'."""
    monkeypatch.delitem(vars(swarms), "__version__", raising=False)

    def _raise(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", _raise)

    assert swarms.__getattr__("__version__") == "unknown"

    monkeypatch.undo()
    vars(swarms).pop("__version__", None)


def test_lazy_agent_import_does_not_load_unneeded_structures():
    """Importing Agent must not load multi-agent structures like AdvisorSwarm or GraphWorkflow."""
    code = (
        "import sys\n"
        "from swarms import Agent\n"
        "loaded = set(sys.modules)\n"
        "unneeded = [\n"
        "    'swarms.structs.advisor_swarm',\n"
        "    'swarms.structs.agent_rearrange',\n"
        "    'swarms.structs.graph_workflow',\n"
        "    'swarms.structs.groupchat',\n"
        "    'swarms.structs.heavy_swarm',\n"
        "    'swarms.structs.hiearchical_swarm',\n"
        "    'swarms.structs.majority_voting',\n"
        "    'swarms.structs.mixture_of_agents',\n"
        "    'swarms.structs.sequential_workflow',\n"
        "]\n"
        "found = [m for m in unneeded if m in loaded]\n"
        "assert not found, f'Unneeded structures loaded: {found}'\n"
        "struct_modules = [m for m in loaded if m.startswith('swarms.structs.')]\n"
        "assert len(struct_modules) <= 15, f'Too many structs loaded: {len(struct_modules)}: {struct_modules}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"stdout: {result.stdout}\nstderr: {result.stderr}"


def test_every_public_name_in_swarms_all_resolves():
    """Every name declared in swarms.__all__ must resolve to an object."""
    assert len(swarms.__all__) >= 160
    for name in swarms.__all__:
        attr = getattr(swarms, name)
        assert attr is not None, f"swarms.{name} resolved to None"


def test_every_public_name_in_structs_all_resolves():
    """Every name declared in swarms.structs.__all__ must resolve to an object."""
    import swarms.structs

    assert len(swarms.structs.__all__) >= 70
    for name in swarms.structs.__all__:
        attr = getattr(swarms.structs, name)
        assert (
            attr is not None
        ), f"swarms.structs.{name} resolved to None"


def test_every_public_name_in_prompts_all_resolves():
    """Every name declared in swarms.prompts.__all__ must resolve to an object."""
    import swarms.prompts

    assert len(swarms.prompts.__all__) >= 10
    for name in swarms.prompts.__all__:
        attr = getattr(swarms.prompts, name)
        assert (
            attr is not None
        ), f"swarms.prompts.{name} resolved to None"


def test_dir_lists_all_exports_before_access():
    """dir() lists all exports before they are accessed."""
    code = (
        "import swarms, swarms.structs, swarms.prompts\n"
        "assert set(swarms.__all__).issubset(set(dir(swarms)))\n"
        "assert set(swarms.structs.__all__).issubset(set(dir(swarms.structs)))\n"
        "assert set(swarms.prompts.__all__).issubset(set(dir(swarms.prompts)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"stdout: {result.stdout}\nstderr: {result.stderr}"


def test_lazy_attribute_cached_after_first_access():
    """Accessing an export caches it in globals() so subsequent access is direct."""
    import swarms

    assert hasattr(swarms, "GroupChat")
    gc = swarms.GroupChat
    assert "GroupChat" in vars(swarms)
    assert swarms.GroupChat is gc


def test_unknown_attribute_on_subpackages_raises_attribute_error():
    """Accessing nonexistent attributes on structs or prompts raises AttributeError."""
    import swarms.prompts
    import swarms.structs

    with pytest.raises(AttributeError):
        _ = swarms.structs.definitely_nonexistent_struct_attr

    with pytest.raises(AttributeError):
        _ = swarms.prompts.definitely_nonexistent_prompt_attr


def test_subpackages_accessible_as_attributes():
    """Subpackages must be accessible as attributes on the swarms module."""
    for subpkg in [
        "agents",
        "prompts",
        "schemas",
        "structs",
        "telemetry",
        "tools",
        "utils",
    ]:
        mod = getattr(swarms, subpkg)
        assert mod is not None
        assert mod.__name__ == f"swarms.{subpkg}"


def test_direct_import_matches_structs_import():
    """from swarms import X must return the exact same object as from swarms.structs import X."""
    from swarms import Agent as RootAgent
    from swarms.structs import Agent as StructAgent

    assert RootAgent is StructAgent

    from swarms import GroupChat as RootGroupChat
    from swarms.structs import GroupChat as StructGroupChat

    assert RootGroupChat is StructGroupChat


if __name__ == "__main__":
    pytest.main([__file__])
