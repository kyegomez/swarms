from unittest.mock import patch
from swarms.structs.autoscaler import AutoScaler
from swarms.models import OpenAIChat
from swarms.structs import Agent

llm = OpenAIChat()

agent = Agent(
    llm=llm,
    max_loops=2,
    dashboard=True,
)


def test_autoscaler_initialization():
    autoscaler = AutoScaler(
        initial_agents=5,
        scale_up_factor=2,
        idle_threshold=0.1,
        busy_threshold=0.8,
        agent=agent,
    )
    assert isinstance(autoscaler, AutoScaler)
    assert autoscaler.scale_up_factor == 2
    assert autoscaler.idle_threshold == 0.1
    assert autoscaler.busy_threshold == 0.8
    assert len(autoscaler.agents_pool) == 5


def test_autoscaler_add_task():
    autoscaler = AutoScaler(agent=agent)
    autoscaler.add_task("task1")
    assert autoscaler.task_queue.qsize() == 1


def test_autoscaler_scale_up():
    autoscaler = AutoScaler(
        initial_agents=5, scale_up_factor=2, agent=agent
    )
    autoscaler.scale_up()
    assert len(autoscaler.agents_pool) == 10


def test_autoscaler_scale_down():
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    autoscaler.scale_down()
    assert len(autoscaler.agents_pool) == 4


@patch("swarms.swarms.AutoScaler.scale_up")
@patch("swarms.swarms.AutoScaler.scale_down")
def test_autoscaler_monitor_and_scale(mock_scale_down, mock_scale_up):
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    autoscaler.add_task("task1")
    autoscaler.monitor_and_scale()
    mock_scale_up.assert_called_once()
    mock_scale_down.assert_called_once()


@patch("swarms.swarms.AutoScaler.monitor_and_scale")
@patch("swarms.swarms.agent.run")
def test_autoscaler_start(mock_run, mock_monitor_and_scale):
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    autoscaler.add_task("task1")
    autoscaler.start()
    mock_run.assert_called_once()
    mock_monitor_and_scale.assert_called_once()


def test_autoscaler_del_agent():
    autoscaler = AutoScaler(initial_agents=5, agent=agent)
    autoscaler.del_agent()
    assert len(autoscaler.agents_pool) == 4
