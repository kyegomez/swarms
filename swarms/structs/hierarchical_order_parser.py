import json
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import TypeAdapter, ValidationError

from swarms.schemas.hs_schemas import (
    HierarchicalOrder,
    OrderBatch,
    SwarmSpec,
)

_MISSING = object()

_ORDERS_ADAPTER = TypeAdapter(List[HierarchicalOrder])

_ORDERS_TOKEN = '"orders"'
_ORDERS_TOKEN_B = b'"orders"'

_JSON_BYTES = (bytes, bytearray)


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


def _json_object(
    value: Union[str, bytes, bytearray],
) -> Optional[Dict[str, Any]]:
    """Decode *value* only when it can hold orders."""
    if isinstance(value, str):
        if _ORDERS_TOKEN not in value:
            return None
    elif isinstance(value, _JSON_BYTES):
        if _ORDERS_TOKEN_B not in value:
            return None
    else:
        return None
    try:
        decoded = json.loads(value)
    except ValueError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _from_function(
    function: Any,
) -> Optional[Tuple[Any, List[HierarchicalOrder]]]:
    """Pull a plan out of an OpenAI-style ``function`` object."""
    if not isinstance(function, dict):
        return None
    arguments = function.get("arguments", _MISSING)
    if arguments is _MISSING:
        return None
    if isinstance(arguments, dict):
        payload = arguments
    else:
        payload = _json_object(arguments)
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
