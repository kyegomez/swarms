import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

# Load Conversation without importing the full swarms package
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
sys.modules.setdefault("swarms.structs", ModuleType("swarms.structs"))
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


class OldConversation(Conversation):
    def return_history_as_string(self):
        formatted = [
            f"{m['role']}: {m['content']}"
            for m in self.conversation_history
        ]
        return "\n\n".join(formatted)

    def get_str(self):
        return self.return_history_as_string()


def measure(conv_cls, messages=50, loops=1000):
    conv = conv_cls(token_count=False)
    for i in range(messages):
        conv.add("user", f"msg{i}")
    start = time.perf_counter()
    for _ in range(loops):
        conv.get_str()
    return time.perf_counter() - start


def test_cache_perf_improvement():
    old_time = measure(OldConversation)
    new_time = measure(Conversation)
    assert old_time / new_time >= 2
