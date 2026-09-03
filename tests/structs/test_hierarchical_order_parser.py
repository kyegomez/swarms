import json

import pytest

from swarms.schemas.hs_schemas import (
    HierarchicalOrder,
    OrderBatch,
    SwarmSpec,
)
from swarms.structs.hierarchical_order_parser import parse_orders


def _spec(**overrides):
    payload = {
        "plan": "do it",
        "orders": [{"agent_name": "Worker", "task": "Do the work"}],
    }
    payload.update(overrides)
    return payload


def test_swarm_spec_is_returned_as_is():
    spec = SwarmSpec(
        plan="do it",
        orders=[
            HierarchicalOrder(agent_name="Worker", task="Do the work")
        ],
    )
    plan, orders = parse_orders(spec)
    assert plan == "do it"
    assert orders is spec.orders


def test_order_batch_returns_orders_without_a_plan():
    batch = OrderBatch(
        orders=[
            HierarchicalOrder(agent_name="Worker", task="Do the work")
        ]
    )
    plan, orders = parse_orders(batch)
    assert plan is None
    assert orders is batch.orders


def test_dict_with_orders_only():
    plan, orders = parse_orders({"orders": _spec()["orders"]})
    assert plan is None
    assert orders[0].agent_name == "Worker"


def test_json_with_orders_only():
    plan, orders = parse_orders(
        json.dumps({"orders": _spec()["orders"]})
    )
    assert plan is None
    assert orders[0].task == "Do the work"


def test_dict_with_plan_and_orders():
    plan, orders = parse_orders(_spec())
    assert plan == "do it"
    assert orders[0].agent_name == "Worker"
    assert orders[0].task == "Do the work"


def test_json_string():
    plan, orders = parse_orders(json.dumps(_spec()))
    assert plan == "do it"
    assert orders[0].agent_name == "Worker"


def test_json_bytes():
    plan, orders = parse_orders(json.dumps(_spec()).encode())
    assert plan == "do it"
    assert orders[0].task == "Do the work"


def test_double_encoded_json_string():
    plan, orders = parse_orders(json.dumps(json.dumps(_spec())))
    assert plan == "do it"
    assert len(orders) == 1


def test_empty_orders_are_valid():
    plan, orders = parse_orders(_spec(orders=[]))
    assert plan == "do it"
    assert orders == []


def test_orders_that_are_already_models_are_not_rebuilt():
    existing = [
        HierarchicalOrder(agent_name="Worker", task="Do the work")
    ]
    plan, orders = parse_orders({"plan": "do it", "orders": existing})
    assert plan == "do it"
    assert orders is existing


def test_direct_tool_call_list_with_json_arguments():
    output = [
        {
            "id": "c1",
            "type": "function",
            "function": {
                "name": "handoff",
                "arguments": json.dumps(_spec()),
            },
        }
    ]
    plan, orders = parse_orders(output)
    assert plan == "do it"
    assert orders[0].agent_name == "Worker"


def test_direct_tool_call_list_with_dict_arguments():
    output = [
        {
            "function": {
                "name": "handoff",
                "arguments": _spec(),
            }
        }
    ]
    plan, orders = parse_orders(output)
    assert plan == "do it"
    assert orders[0].task == "Do the work"


def test_conversation_turn_with_tool_call_in_content():
    output = [
        {
            "role": "assistant",
            "content": [
                {
                    "function": {
                        "name": "handoff",
                        "arguments": json.dumps(_spec()),
                    }
                }
            ],
        }
    ]
    plan, orders = parse_orders(output)
    assert plan == "do it"
    assert orders[0].agent_name == "Worker"


def test_bare_spec_inside_a_list():
    plan, orders = parse_orders([_spec()])
    assert plan == "do it"
    assert orders[0].agent_name == "Worker"


def test_single_tool_call_dict_not_wrapped_in_a_list():
    output = {
        "function": {
            "name": "handoff",
            "arguments": _spec(),
        }
    }
    plan, orders = parse_orders(output)
    assert plan == "do it"
    assert orders[0].agent_name == "Worker"


def test_skips_argument_blobs_that_cannot_hold_a_plan():
    output = [
        {
            "function": {
                "name": "other",
                "arguments": json.dumps({"foo": "bar"}),
            }
        },
        {
            "function": {
                "name": "handoff",
                "arguments": json.dumps(_spec()),
            }
        },
    ]
    plan, orders = parse_orders(output)
    assert plan == "do it"
    assert orders[0].agent_name == "Worker"


def test_invalid_json_string_raises_value_error():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_orders("{not json")


def test_dict_missing_orders_raises_value_error():
    with pytest.raises(ValueError, match="Missing 'orders'"):
        parse_orders({"plan": "do it"})


def test_empty_list_raises_value_error():
    with pytest.raises(ValueError, match="Unable to parse"):
        parse_orders([])


def test_unexpected_type_raises_value_error():
    with pytest.raises(ValueError, match="Unexpected output format"):
        parse_orders(42)


def test_malformed_order_raises_value_error():
    with pytest.raises(ValueError, match="Invalid orders"):
        parse_orders(
            {"plan": "do it", "orders": [{"agent_name": "Worker"}]}
        )


def test_hierarchical_swarm_delegates_to_the_parser():
    from swarms.structs.hiearchical_swarm import HierarchicalSwarm

    class Director:
        agent_name = "Director"
        output_type = "final"

    class Worker:
        agent_name = "Worker"
        output_type = "final"

    swarm = HierarchicalSwarm(
        director=Director(),
        agents=[Worker()],
        autosave=False,
        planning_enabled=False,
        add_collaboration_prompt=False,
    )
    plan, orders = swarm.parse_orders(
        [
            {
                "function": {
                    "name": "handoff",
                    "arguments": _spec(),
                }
            }
        ]
    )
    assert plan == "do it"
    assert orders[0].agent_name == "Worker"
