import hashlib
import uuid
import types


def _dummy_completion(*args, **kwargs):
    # Return an object that is both iterable (for streaming tests)
    # and has a `.choices` attribute for legacy call sites.
    class _Resp:
        def __init__(self):
            self.choices = [{"message": {"content": "DUMMY_RESPONSE"}}]

        def __iter__(self):
            # yield a couple of chunk-like dicts
            yield {"choices": [{"delta": {"content": "DUMMY_CHUNK_1"}}]}
            yield {"choices": [{"delta": {"content": "DUMMY_CHUNK_2"}}]}

    return _Resp()


def pytest_configure(config):
    # Patch model token lookup to avoid litellm mapping errors
    try:
        import swarms.utils.check_all_model_max_tokens as cam

        cam.get_single_model_max_tokens = lambda name: 4096
    except Exception:
        pass

    # Make machine id non-deterministic for telemetry tests
    try:
        import swarms.telemetry.main as telemetry

        telemetry.get_machine_id = lambda: hashlib.sha256(uuid.uuid4().hex.encode()).hexdigest()
    except Exception:
        pass

    # Stub litellm completion to avoid network/auth calls during tests
    try:
        import litellm

        litellm.completion = _dummy_completion
    except Exception:
        pass

    # Also patch local wrapper import used across the package
    try:
        import swarms.utils.litellm_wrapper as lw

        lw.completion = _dummy_completion
    except Exception:
        pass
