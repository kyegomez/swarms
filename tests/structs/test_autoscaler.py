import os

from dotenv import load_dotenv
from pytest import patch

from swarms.models import OpenAIChat
from swarms.structs import Agent
from swarms.structs.autoscaler import AutoScaler

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
llm = OpenAIChat(
    temperature=0.5,
    openai_api_key=api_key,
)
agent = Agent(llm=llm, max_loops=1)


def test_autoscaler_init():
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    assert autoscaler.initial_agents == 5
    assert autoscaler.scale_up_factor == 1
    assert autoscaler.idle_threshold == 0.2
    assert autoscaler.busy_threshold == 0.7
    assert autoscaler.autoscale == True
    assert autoscaler.min_agents == 1
    assert autoscaler.max_agents == 5
    assert autoscaler.custom_scale_strategy == None
    assert len(autoscaler.agents_pool) == 5
    assert autoscaler.task_queue.empty() == True


def test_autoscaler_add_task():
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    autoscaler.add_task("task1")
    assert autoscaler.task_queue.empty() == False


def test_autoscaler_run():
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    out = autoscaler.run(
        agent.id,
        "Generate a 10,000 word blog on health and wellness.",
    )
    assert (
        out == "Generate a 10,000 word blog on health and wellness."
    )


def test_autoscaler_add_agent():
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    autoscaler.add_agent(agent)
    assert len(autoscaler.agents_pool) == 6


def test_autoscaler_remove_agent():
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    autoscaler.remove_agent(agent)
    assert len(autoscaler.agents_pool) == 4


def test_autoscaler_get_agent():
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    agent = autoscaler.get_agent()
    assert isinstance(agent, Agent)


def test_autoscaler_get_agent_by_id():
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    agent = autoscaler.get_agent_by_id(agent.id)
    assert isinstance(agent, Agent)


def test_autoscaler_get_agent_by_id_not_found():
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    agent = autoscaler.get_agent_by_id("fake_id")
    assert agent == None


@patch("swarms.swarms.Agent.is_healthy")
def test_autoscaler_check_agent_health(mock_is_healthy):
    mock_is_healthy.side_effect = [False, True, True, True, True]
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    autoscaler.check_agent_health()
    assert mock_is_healthy.call_count == 5


def test_autoscaler_balance_load():
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    autoscaler.add_task("task1")
    autoscaler.add_task("task2")
    autoscaler.balance_load()
    assert autoscaler.task_queue.empty()


def test_autoscaler_set_scaling_strategy():
    autoscaler = AutoScaler(initial_agents=5, agent=agent)

    def strategy(x, y):
        return x - y

    autoscaler.set_scaling_strategy(strategy)
    assert autoscaler.custom_scale_strategy == strategy


def test_autoscaler_execute_scaling_strategy():
    autoscaler = AutoScaler(initial_agents=5, agent=agent)

    def strategy(x, y):
        return x - y

    autoscaler.set_scaling_strategy(strategy)
    autoscaler.add_task("task1")
    autoscaler.execute_scaling_strategy()
    assert len(autoscaler.agents_pool) == 4


def test_autoscaler_report_agent_metrics():
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    metrics = autoscaler.report_agent_metrics()
    assert set(metrics.keys()) == {
        "completion_time",
        "success_rate",
        "error_rate",
    }


@patch("swarms.swarms.AutoScaler.report_agent_metrics")
def test_autoscaler_report(mock_report_agent_metrics):
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    autoscaler.report()
    mock_report_agent_metrics.assert_called_once()


@patch("builtins.print")
def test_autoscaler_print_dashboard(mock_print):
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    autoscaler.print_dashboard()
    mock_print.assert_called()
