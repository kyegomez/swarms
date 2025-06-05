import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# Load Conversation without importing swarms package to avoid optional deps
CONV_PATH = (
    Path(__file__).resolve().parents[2]
    / "swarms"
    / "structs"
    / "conversation.py"
)
spec = importlib.util.spec_from_file_location(
    "swarms.structs.conversation", CONV_PATH
)
conversation = importlib.util.module_from_spec(spec)
sys.modules.setdefault("swarms", ModuleType("swarms"))
structs_module = ModuleType("swarms.structs")
sys.modules.setdefault("swarms.structs", structs_module)
# Minimal BaseStructure to satisfy Conversation import
base_module = ModuleType("swarms.structs.base_structure")
base_module.BaseStructure = object
sys.modules["swarms.structs.base_structure"] = base_module
utils_root = ModuleType("swarms.utils")
any_to_str_mod = ModuleType("swarms.utils.any_to_str")
formatter_mod = ModuleType("swarms.utils.formatter")
token_mod = ModuleType("swarms.utils.litellm_tokenizer")
any_to_str_mod.any_to_str = lambda x: str(x)
formatter_mod.formatter = type(
    "Formatter", (), {"print_panel": lambda *a, **k: None}
)()
token_mod.count_tokens = lambda s: len(str(s).split())
sys.modules["swarms.utils"] = utils_root
sys.modules["swarms.utils.any_to_str"] = any_to_str_mod
sys.modules["swarms.utils.formatter"] = formatter_mod
sys.modules["swarms.utils.litellm_tokenizer"] = token_mod
spec.loader.exec_module(conversation)
Conversation = conversation.Conversation


def test_history_cache_updates_incrementally():
    conv = Conversation(token_count=False)
    conv.add("user", "Hello")
    first_cache = conv.get_str()
    assert first_cache == "user: Hello"
    conv.add("assistant", "Hi")
    second_cache = conv.get_str()
    assert second_cache.endswith("assistant: Hi")
    assert conv._cache_index == len(conv.conversation_history)
    # Ensure cache reused when no new messages
    assert conv.get_str() is second_cache
