import json
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import TypeAdapter, ValidationError

from swarms.schemas.hs_schemas import (
    HierarchicalOrder,
    OrderBatch,
    SwarmSpec,
)
from swarms.utils.str_to_dict import tool_call_arguments

_MISSING = object()

_ORDERS_ADAPTER = TypeAdapter(List[HierarchicalOrder])


def _orders(raw: Any) -> List[HierarchicalOrder]:
    """Validate *raw* as orders, skipping work when they are already models."""
    if raw == []:
        return []
    if isinstance(raw, list) and raw:
        if isinstance(raw[0], HierarchicalOrder) and all(
            isinstance(item, HierarchicalOrder) for item in raw
        ):
            return raw
    try:
        return _ORDERS_ADAPTER.validate_python(raw)
    except ValidationError as error:
        raise ValueError(
            f"Invalid orders in director output: {raw!r}"
        ) from error


def _plan_and_orders(
    payload: Dict[str, Any],
) -> Optional[Tuple[Any, List[HierarchicalOrder]]]:
    """Return an optional plan and orders, or None when orders are absent."""
    orders = payload.get("orders", _MISSING)
    if orders is _MISSING:
        return None
    return payload.get("plan"), _orders(orders)


def _from_function(
    function: Any,
) -> Optional[Tuple[Any, List[HierarchicalOrder]]]:
    """Pull a plan out of an OpenAI-style ``function`` object."""
    if not isinstance(function, dict):
        return None
    payload = tool_call_arguments({"function": function})
    if payload is None:
        return None
    return _plan_and_orders(payload)


def _from_blocks(
    items: List[Any],
) -> Optional[Tuple[Any, List[HierarchicalOrder]]]:
    """Scan tool-call blocks — or conversation turns holding them — for a plan."""
    for item in items:
        if not isinstance(item, dict):
            continue

        parsed = _from_function(item.get("function"))
        if parsed is not None:
            return parsed

        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                parsed = _from_function(block.get("function"))
                if parsed is not None:
                    return parsed
            continue

        parsed = _plan_and_orders(item)
        if parsed is not None:
            return parsed
    return None


def _loads(output: Union[str, bytes, bytearray]) -> Any:
    """``json.loads``, once more if the director double-encoded the payload."""
    try:
        decoded = json.loads(output)
    except ValueError as error:
        raise ValueError(
            f"Director output is not valid JSON: {output!r}"
        ) from error
    if isinstance(decoded, str):
        try:
            return json.loads(decoded)
        except ValueError as error:
            raise ValueError(
                f"Director output is not valid JSON: {output!r}"
            ) from error
    return decoded


def parse_orders(output: Any) -> Tuple[Any, List[HierarchicalOrder]]:
    """Extract hierarchical orders and an optional legacy plan.

    Args:
        output: Raw director output as an :class:`OrderBatch`, legacy
            :class:`SwarmSpec`, JSON, mapping, tool call, or conversation.

    Returns:
        ``(plan, orders)`` where plan is ``None`` for order batches.

    Raises:
        ValueError: If the output shape is unrecognised, is not valid JSON,
            or carries no ``plan``/``orders`` pair.
    """
    if isinstance(output, SwarmSpec):
        return output.plan, output.orders

    if isinstance(output, OrderBatch):
        return None, output.orders

    if isinstance(output, (str, bytes, bytearray)):
        output = _loads(output)

    if isinstance(output, dict):
        parsed = _plan_and_orders(output)
        if parsed is not None:
            return parsed
        parsed = _from_blocks([output])
        if parsed is not None:
            return parsed
        raise ValueError(
            f"Missing 'orders' in director output: {output}"
        )

    if isinstance(output, list):
        parsed = _from_blocks(output)
        if parsed is None:
            raise ValueError(
                f"Unable to parse orders from director output: {output}"
            )
        return parsed

    raise ValueError(
        f"Unexpected output format from director: {type(output)}"
    )
