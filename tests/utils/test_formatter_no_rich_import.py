import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])


def test_formatter_ast_no_top_level_rich():
    """Verify that swarms/utils/formatter.py does not import rich at module level."""
    formatter_path = (
        Path(REPO_ROOT) / "swarms" / "utils" / "formatter.py"
    )
    tree = ast.parse(formatter_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(
                    "rich"
                ), f"Top-level 'import {alias.name}' found in formatter.py"
        elif isinstance(node, ast.ImportFrom):
            assert not (
                node.module or ""
            ).startswith(
                "rich"
            ), f"Top-level 'from {node.module} import ...' found in formatter.py"


def test_import_formatter_does_not_load_rich():
    """Importing formatter module in isolation must not pull rich into sys.modules."""
    code = (
        "import sys\n"
        "import importlib.util\n"
        "from pathlib import Path\n"
        "formatter_path = Path(sys.argv[1])\n"
        "spec = importlib.util.spec_from_file_location('formatter', formatter_path)\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "assert 'rich' not in sys.modules\n"
        "spec.loader.exec_module(mod)\n"
        "assert 'rich' not in sys.modules, f'rich was loaded: {sys.modules.get(\"rich\")}'\n"
        "fmt = mod.Formatter(md=False)\n"
        "assert 'rich' not in sys.modules, 'instantiating Formatter loaded rich prematurely'\n"
        "fmt._print_panel('test', title='test')\n"
        "assert 'rich' in sys.modules, 'printing did not load rich'\n"
    )
    formatter_path = str(
        Path(REPO_ROOT) / "swarms" / "utils" / "formatter.py"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, formatter_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, f"Subprocess failed with stderr:\n{proc.stderr}"


def test_formatter_spinner_getattr():
    """Verify swarms.utils.formatter.spinner backward compatibility."""
    formatter_path = (
        Path(REPO_ROOT) / "swarms" / "utils" / "formatter.py"
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "formatter_test_spinner", formatter_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    spinner = mod.spinner
    from rich.spinner import Spinner

    assert isinstance(spinner, Spinner)
