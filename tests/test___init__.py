"""
Package import-time contract.

`import swarms` must be side-effect-light: heavyweight provider/client
libraries (litellm pulls in openai and fetches its model-cost map from the
network; mcp pulls in the whole server stack) load on first *use*, not on
import (#1754, #1739). Run in a subprocess so an already-polluted
sys.modules in the test runner cannot mask a regression.
"""

import json
import subprocess
import sys


def _probe(code: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            "SWARMS_TELEMETRY_ON": "false",
            # Keep the probe offline: litellm must not fetch its
            # model-cost map from the network inside a unit test.
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "PATH": "",
        },
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_import_swarms_defers_litellm_and_mcp():
    """Neither litellm nor mcp (nor openai) may load at import time."""
    result = _probe(
        "import json, sys\n"
        "import swarms\n"
        "print(json.dumps({m: (m in sys.modules)"
        " for m in ('litellm', 'mcp', 'openai')}))\n"
    )
    assert result == {
        "litellm": False,
        "mcp": False,
        "openai": False,
    }, f"eagerly imported: {[m for m, v in result.items() if v]}"


def test_litellm_binds_on_first_llm_construction():
    """Deferral must not break first use: constructing the LLM loads litellm."""
    result = _probe(
        "import json, sys\n"
        "from swarms.utils.litellm_wrapper import LiteLLM\n"
        "before = 'litellm' in sys.modules\n"
        "LiteLLM(model_name='gpt-4o-mini')\n"
        "after = 'litellm' in sys.modules\n"
        "print(json.dumps({'before': before, 'after': after}))\n"
    )
    assert result == {"before": False, "after": True}
