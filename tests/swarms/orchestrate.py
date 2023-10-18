import numpy as np
from swarms.swarms.orchestrate import Orchestrator, Worker
import chromadb


def test_orchestrator_initialization():
    orchestrator = Orchestrator(agent=Worker, agent_list=[Worker] * 5, task_queue=[])
    assert isinstance(orchestrator, Orchestrator)
    assert orchestrator.agents.qsize() == 5
    assert orchestrator.task_queue.qsize() == 0


def test_orchestrator_assign_task():
    orchestrator = Orchestrator(agent=Worker, agent_list=[Worker] * 5, task_queue=[])
    orchestrator.assign_task(1, {"content": "task1"})
    assert orchestrator.task_queue.qsize() == 1


def test_orchestrator_embed():
    orchestrator = Orchestrator(agent=Worker, agent_list=[Worker] * 5, task_queue=[])
    result = orchestrator.embed("Hello, world!", "api_key", "model_name")
    assert isinstance(result, np.ndarray)


def test_orchestrator_retrieve_results():
    orchestrator = Orchestrator(agent=Worker, agent_list=[Worker] * 5, task_queue=[])
    result = orchestrator.retrieve_results(1)
    assert isinstance(result, list)


def test_orchestrator_update_vector_db():
    orchestrator = Orchestrator(agent=Worker, agent_list=[Worker] * 5, task_queue=[])
    data = {"vector": np.array([1, 2, 3]), "task_id": 1}
    orchestrator.update_vector_db(data)
    assert orchestrator.collection.count() == 1


def test_orchestrator_get_vector_db():
    orchestrator = Orchestrator(agent=Worker, agent_list=[Worker] * 5, task_queue=[])
    result = orchestrator.get_vector_db()
    assert isinstance(result, chromadb.Collection)


def test_orchestrator_append_to_db():
    orchestrator = Orchestrator(agent=Worker, agent_list=[Worker] * 5, task_queue=[])
    orchestrator.append_to_db("Hello, world!")
    assert orchestrator.collection.count() == 1


def test_orchestrator_run():
    orchestrator = Orchestrator(agent=Worker, agent_list=[Worker] * 5, task_queue=[])
    result = orchestrator.run("Write a short story.")
    assert isinstance(result, list)


def test_orchestrator_chat():
    orchestrator = Orchestrator(agent=Worker, agent_list=[Worker] * 5, task_queue=[])
    orchestrator.chat(1, 2, "Hello, Agent 2!")
    assert orchestrator.collection.count() == 1


def test_orchestrator_add_agents():
    orchestrator = Orchestrator(agent=Worker, agent_list=[Worker] * 5, task_queue=[])
    orchestrator.add_agents(5)
    assert orchestrator.agents.qsize() == 10


def test_orchestrator_remove_agents():
    orchestrator = Orchestrator(agent=Worker, agent_list=[Worker] * 5, task_queue=[])
    orchestrator.remove_agents(3)
    assert orchestrator.agents.qsize() == 2
