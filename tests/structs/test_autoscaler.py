import os
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

from swarms.models import OpenAIChat
from swarms.structs import Agent
from swarms.structs.autoscaler import AutoScaler

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
llm = OpenAIChat(
    temperature=0.5,
    openai_api_key=api_key,
)
global_agent = Agent(llm=llm, max_loops=1)


def test_autoscaler_init():
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    assert autoscaler.initial_agents == 5
    assert autoscaler.scale_up_factor == 1
    assert autoscaler.idle_threshold == 0.2
    assert autoscaler.busy_threshold == 0.7
    assert autoscaler.autoscale is True
    assert autoscaler.min_agents == 1
    assert autoscaler.max_agents == 5
    assert autoscaler.custom_scale_strategy is None
    assert len(autoscaler.agents_pool) == 5
    assert autoscaler.task_queue.empty() is True


def test_autoscaler_add_task():
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    autoscaler.add_task("task1")
    assert autoscaler.task_queue.empty() is False


def test_autoscaler_run():
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    out = autoscaler.run(
        global_agent.id,
        "Generate a 10,000 word blog on health and wellness.",
    )
    assert (
        out == "Generate a 10,000 word blog on health and wellness."
    )


def test_autoscaler_add_agent():
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    autoscaler.add_agent(global_agent)
    assert len(autoscaler.agents_pool) == 6


def test_autoscaler_remove_agent():
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    autoscaler.remove_agent(global_agent)
    assert len(autoscaler.agents_pool) == 4


def test_autoscaler_get_agent():
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    agent = autoscaler.get_agent()
    assert isinstance(agent, Agent)


def test_autoscaler_get_agent_by_id():
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    agent = autoscaler.get_agent_by_id(global_agent.id)
    assert isinstance(agent, Agent)


def test_autoscaler_get_agent_by_id_not_found():
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    agent = autoscaler.get_agent_by_id("fake_id")
    assert agent is None


@patch("swarms.swarms.Agent.is_healthy")
def test_autoscaler_check_agent_health(mock_is_healthy):
    mock_is_healthy.side_effect = [False, True, True, True, True]
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    autoscaler.check_agent_health()
    assert mock_is_healthy.call_count == 5


def test_autoscaler_balance_load():
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    autoscaler.add_task("task1")
    autoscaler.add_task("task2")
    autoscaler.balance_load()
    assert autoscaler.task_queue.empty()


def test_autoscaler_set_scaling_strategy():
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)

    def strategy(x, y):
        return x - y

    autoscaler.set_scaling_strategy(strategy)
    assert autoscaler.custom_scale_strategy == strategy


def test_autoscaler_execute_scaling_strategy():
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)

    def strategy(x, y):
        return x - y

    autoscaler.set_scaling_strategy(strategy)
    autoscaler.add_task("task1")
    autoscaler.execute_scaling_strategy()
    assert len(autoscaler.agents_pool) == 4


def test_autoscaler_report_agent_metrics():
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    metrics = autoscaler.report_agent_metrics()
    assert set(metrics.keys()) == {
        "completion_time",
        "success_rate",
        "error_rate",
    }


@patch("swarms.swarms.AutoScaler.report_agent_metrics")
def test_autoscaler_report(mock_report_agent_metrics):
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    autoscaler.report()
    mock_report_agent_metrics.assert_called_once()


@patch("builtins.print")
def test_autoscaler_print_dashboard(mock_print):
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    autoscaler.print_dashboard()
    mock_print.assert_called()


@patch("swarms.structs.autoscaler.logging")
def test_check_agent_health_all_healthy(mock_logging):
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    for agent in autoscaler.agents_pool:
        agent.is_healthy = MagicMock(return_value=True)
    autoscaler.check_agent_health()
    mock_logging.warning.assert_not_called()


@patch("swarms.structs.autoscaler.logging")
def test_check_agent_health_some_unhealthy(mock_logging):
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    for i, agent in enumerate(autoscaler.agents_pool):
        agent.is_healthy = MagicMock(return_value=(i % 2 == 0))
    autoscaler.check_agent_health()
    assert mock_logging.warning.call_count == 2


@patch("swarms.structs.autoscaler.logging")
def test_check_agent_health_all_unhealthy(mock_logging):
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    for agent in autoscaler.agents_pool:
        agent.is_healthy = MagicMock(return_value=False)
    autoscaler.check_agent_health()
    assert mock_logging.warning.call_count == 5


@patch("swarms.structs.autoscaler.Agent")
def test_add_agent(mock_agent):
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    initial_count = len(autoscaler.agents_pool)
    autoscaler.add_agent()
    assert len(autoscaler.agents_pool) == initial_count + 1
    mock_agent.assert_called_once()


@patch("swarms.structs.autoscaler.Agent")
def test_remove_agent(mock_agent):
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    initial_count = len(autoscaler.agents_pool)
    autoscaler.remove_agent()
    assert len(autoscaler.agents_pool) == initial_count - 1


@patch("swarms.structs.autoscaler.AutoScaler.add_agent")
@patch("swarms.structs.autoscaler.AutoScaler.remove_agent")
def test_scale(mock_remove_agent, mock_add_agent):
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    autoscaler.scale(10)
    assert mock_add_agent.call_count == 5
    assert mock_remove_agent.call_count == 0

    mock_add_agent.reset_mock()
    mock_remove_agent.reset_mock()

    autoscaler.scale(3)
    assert mock_add_agent.call_count == 0
    assert mock_remove_agent.call_count == 2


def test_add_task_success():
    autoscaler = AutoScaler(initial_agents=5)
    initial_queue_size = autoscaler.task_queue.qsize()
    autoscaler.add_task("test_task")
    assert autoscaler.task_queue.qsize() == initial_queue_size + 1


@patch("swarms.structs.autoscaler.queue.Queue.put")
def test_add_task_exception(mock_put):
    mock_put.side_effect = Exception("test error")
    autoscaler = AutoScaler(initial_agents=5)
    with pytest.raises(Exception) as e:
        autoscaler.add_task("test_task")
    assert str(e.value) == "test error"


def test_autoscaler_initialization():
    autoscaler = AutoScaler(
        initial_agents=5,
        scale_up_factor=2,
        idle_threshold=0.1,
        busy_threshold=0.8,
        agent=global_agent,
    )
    assert isinstance(autoscaler, AutoScaler)
    assert autoscaler.scale_up_factor == 2
    assert autoscaler.idle_threshold == 0.1
    assert autoscaler.busy_threshold == 0.8
    assert len(autoscaler.agents_pool) == 5


def test_autoscaler_add_task_2():
    autoscaler = AutoScaler(agent=global_agent)
    autoscaler.add_task("task1")
    assert autoscaler.task_queue.qsize() == 1


def test_autoscaler_scale_up():
    autoscaler = AutoScaler(
        initial_agents=5, scale_up_factor=2, agent=global_agent
    )
    autoscaler.scale_up()
    assert len(autoscaler.agents_pool) == 10


def test_autoscaler_scale_down():
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    autoscaler.scale_down()
    assert len(autoscaler.agents_pool) == 4


@patch("swarms.swarms.AutoScaler.scale_up")
@patch("swarms.swarms.AutoScaler.scale_down")
def test_autoscaler_monitor_and_scale(mock_scale_down, mock_scale_up):
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    autoscaler.add_task("task1")
    autoscaler.monitor_and_scale()
    mock_scale_up.assert_called_once()
    mock_scale_down.assert_called_once()


@patch("swarms.swarms.AutoScaler.monitor_and_scale")
@patch("swarms.swarms.agent.run")
def test_autoscaler_start(mock_run, mock_monitor_and_scale):
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    autoscaler.add_task("task1")
    autoscaler.start()
    mock_run.assert_called_once()
    mock_monitor_and_scale.assert_called_once()


def test_autoscaler_del_agent():
    autoscaler = AutoScaler(initial_agents=5, agent=global_agent)
    autoscaler.del_agent()
    assert len(autoscaler.agents_pool) == 4
