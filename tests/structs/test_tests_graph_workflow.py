import pytest

from swarms.structs.graph_workflow import GraphWorkflow


@pytest.fixture
def graph_workflow():
    return GraphWorkflow()


def test_init(graph_workflow):
    assert graph_workflow.graph == {}
    assert graph_workflow.entry_point is None


def test_add(graph_workflow):
    graph_workflow.add("node1", "value1")
    assert "node1" in graph_workflow.graph
    assert graph_workflow.graph["node1"]["value"] == "value1"
    assert graph_workflow.graph["node1"]["edges"] == {}


def test_set_entry_point(graph_workflow):
    graph_workflow.add("node1", "value1")
    graph_workflow.set_entry_point("node1")
    assert graph_workflow.entry_point == "node1"


def test_set_entry_point_nonexistent_node(graph_workflow):
    with pytest.raises(ValueError, match="Node does not exist in graph"):
        graph_workflow.set_entry_point("nonexistent")


def test_add_edge(graph_workflow):
    graph_workflow.add("node1", "value1")
    graph_workflow.add("node2", "value2")
    graph_workflow.add_edge("node1", "node2")
    assert "node2" in graph_workflow.graph["node1"]["edges"]


def test_add_edge_nonexistent_node(graph_workflow):
    graph_workflow.add("node1", "value1")
    with pytest.raises(ValueError, match="Node does not exist in graph"):
        graph_workflow.add_edge("node1", "nonexistent")


def test_add_conditional_edges(graph_workflow):
    graph_workflow.add("node1", "value1")
    graph_workflow.add("node2", "value2")
    graph_workflow.add_conditional_edges(
        "node1", "condition1", {"condition_value1": "node2"}
    )
    assert "node2" in graph_workflow.graph["node1"]["edges"]


def test_add_conditional_edges_nonexistent_node(graph_workflow):
    graph_workflow.add("node1", "value1")
    with pytest.raises(ValueError, match="Node does not exist in graph"):
        graph_workflow.add_conditional_edges(
            "node1", "condition1", {"condition_value1": "nonexistent"}
        )


def test_run(graph_workflow):
    graph_workflow.add("node1", "value1")
    graph_workflow.set_entry_point("node1")
    assert graph_workflow.run() == graph_workflow.graph


def test_run_no_entry_point(graph_workflow):
    with pytest.raises(ValueError, match="Entry point not set"):
        graph_workflow.run()
