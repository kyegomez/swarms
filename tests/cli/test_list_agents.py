import sys
import types
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

# Create minimal stub modules to satisfy imports in cli.main
swarms_pkg = types.ModuleType("swarms")
sys.modules["swarms"] = swarms_pkg

agents_pkg = types.ModuleType("swarms.agents")
sys.modules["swarms.agents"] = agents_pkg

auto_module = types.ModuleType("swarms.agents.auto_generate_swarm_config")
auto_module.generate_swarm_config = lambda *a, **k: None
sys.modules["swarms.agents.auto_generate_swarm_config"] = auto_module

create_module = types.ModuleType("swarms.agents.create_agents_from_yaml")
create_module.create_agents_from_yaml = lambda *a, **k: []
sys.modules["swarms.agents.create_agents_from_yaml"] = create_module

structs_pkg = types.ModuleType("swarms.structs")
sys.modules["swarms.structs"] = structs_pkg

agent_registry_module = types.ModuleType("swarms.structs.agent_registry")
class DummyRegistry:
    def __init__(self, *_, **__):
        self.agents = []
    def add_many(self, agents):
        self.agents.extend(agents)
    def list_agents(self):
        return [a.agent_name for a in self.agents]
agent_registry_module.AgentRegistry = DummyRegistry
sys.modules["swarms.structs.agent_registry"] = agent_registry_module

cli_pkg = types.ModuleType("swarms.cli")
sys.modules["swarms.cli"] = cli_pkg

onboarding_module = types.ModuleType("swarms.cli.onboarding_process")
class OnboardingProcess:
    def run(self):
        pass
onboarding_module.OnboardingProcess = OnboardingProcess
sys.modules["swarms.cli.onboarding_process"] = onboarding_module

utils_pkg = types.ModuleType("swarms.utils")
sys.modules["swarms.utils"] = utils_pkg

formatter_module = types.ModuleType("swarms.utils.formatter")
class DummyFormatter:
    def print_panel(self, *args, **kwargs):
        pass
formatter_module.formatter = DummyFormatter()
sys.modules["swarms.utils.formatter"] = formatter_module

# Load CLI module
spec = importlib.util.spec_from_file_location("cli_main", Path("swarms/cli/main.py"))
cli_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli_main)


def test_list_agents_command(capsys):
    mock_agent1 = MagicMock(agent_name="Agent1")
    mock_agent2 = MagicMock(agent_name="Agent2")

    with patch.object(cli_main, "create_agents_from_yaml", return_value=[mock_agent1, mock_agent2]):
        testargs = ["swarms", "list-agents", "--yaml-file", "dummy.yaml"]
        with patch.object(sys, "argv", testargs), patch.object(cli_main.os.path, "exists", return_value=True):
            cli_main.main()

    captured = capsys.readouterr()
    assert "Agent1" in captured.out
    assert "Agent2" in captured.out

