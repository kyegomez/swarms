"""Environment loading helpers for Swarms."""

import os

from dotenv import find_dotenv, load_dotenv


def load_swarms_env(*, override: bool = False) -> bool:
    """Load a ``.env`` file by searching upward from the current working directory.

    Also defaults ``LITELLM_LOCAL_MODEL_COST_MAP`` to ``"True"``, so importing
    litellm reads the model-cost map bundled with it instead of downloading
    one. A value already in the environment or the ``.env`` file wins. To use
    litellm's remote map, set the variable to an empty string: litellm treats
    any non-empty value, including ``"False"``, as on.
    """
    dotenv_path = find_dotenv(".env", usecwd=True)
    loaded = (
        load_dotenv(dotenv_path=dotenv_path, override=override)
        if dotenv_path
        else False
    )
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    return loaded
