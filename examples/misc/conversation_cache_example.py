import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

# Load Conversation without importing full swarms package
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

# Demonstrate cached conversation history
conv = Conversation()
conv.add("user", "Hello")
conv.add("assistant", "Hi there!")

print(conv.get_str())
print(conv.get_str())  # reuses cached string

# Timing demo
start = time.perf_counter()
for _ in range(1000):
    conv.get_str()
cached_time = time.perf_counter() - start
print("Cached retrieval:", round(cached_time, 6), "seconds")


# Compare to rebuilding manually
def slow_get():
    formatted = [
        f"{m['role']}: {m['content']}"
        for m in conv.conversation_history
    ]
    return "\n\n".join(formatted)


start = time.perf_counter()
for _ in range(1000):
    slow_get()
slow_time = time.perf_counter() - start
print("Manual join:", round(slow_time, 6), "seconds")
print("Speedup:", round(slow_time / cached_time, 2), "x")
