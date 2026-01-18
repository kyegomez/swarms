# sitecustomize runs very early on Python startup. Use it to apply test-friendly
# shims before package imports during pytest collection.
import hashlib
import uuid

try:
    # Patch litellm completion to avoid network calls during test collection
    import litellm

    def _sc_dummy_completion(*args, **kwargs):
        class _Resp:
            def __init__(self):
                self.choices = [{"message": {"content": "DUMMY_RESPONSE"}}]

            def __iter__(self):
                yield {"choices": [{"delta": {"content": "DUMMY_CHUNK_1"}}]}

        return _Resp()

    litellm.completion = _sc_dummy_completion
except Exception:
    pass

try:
    # Patch internal helpers that run at import-time
    import swarms.utils.check_all_model_max_tokens as cam

    cam.get_single_model_max_tokens = lambda name: 4096
except Exception:
    pass

try:
    import swarms.telemetry.main as telemetry

    telemetry.get_machine_id = lambda: hashlib.sha256(uuid.uuid4().hex.encode()).hexdigest()
except Exception:
    pass

try:
    # Stub MCP client tools to avoid network during pytest collection
    import swarms.tools.mcp_client_tools as mcp_tools

    def _noop_get_tools(*args, **kwargs):
        return []

    mcp_tools.get_tools_for_multiple_mcp_servers = _noop_get_tools
    mcp_tools.get_mcp_tools_sync = _noop_get_tools
except Exception:
    pass
