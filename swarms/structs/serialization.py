import json
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from swarms.utils.loguru_logger import logger


class SerializableMixin:
    """
    Mixin that provides robust serialization of an object's attributes into a dictionary
    representation via the `to_dict()` method.

    Designed for use with structures/classes (e.g., `Agent`, `AgentRearrange`,
    `SwarmRouter`, `BaseStructure`, etc.) that require best-effort object state
    introspection for purposes such as telemetry, logging, caching, debugging,
    or persistence.

    Main Features:
    --------------
    - **Defensive Serialization:** Handles a wide variety of attribute types,
      including those that are not directly JSON serializable.
    - **Callable Support:** Any attribute that is a function or method will be
      represented as a dictionary containing its `"name"` and `"doc"` string,
      rather than attempting to serialize the code object itself.
    - **Nested Object Support:** If an attribute has a `to_dict()` method
      (including mixins or nested user-defined classes), it will recurse
      into that for a structured, hierarchical representation.
    - **JSON-Serializable Pass-Through:** Any attribute value already compatible
      with `json.dumps` (primitives, lists, dicts, etc.) is returned as-is.
    - **Fallback for Unknowns:** Any attribute that cannot be serialized is
      represented as a string `"<Non-serializable: TypeName>"`, where
      `TypeName` is the class name of the problematic object.
    - **Exclusion Support:** Subclasses may define a class attribute
      `_to_dict_exclude` as a tuple of attribute names to skip during
      serialization, avoiding logging or persisting sensitive or redundant fields.

    Example:
    --------
        class Agent(SerializableMixin):
            _to_dict_exclude: Tuple[str, ...] = ("llm",)

        agent = Agent()
        agent.to_dict()

    Implementation Notes:
    --------------------
    - The mixin is designed to be inherited, providing a default `to_dict()`
      for subclasses. It does not interfere with other class behaviors.
    - Extend or override `_serialize_attr` in subclasses for custom logic.
    - Thread safety: Reading attributes is safe, but `to_dict` is not
      transactionally synchronized; races are possible if used concurrently
      with `__dict__` mutation.

    Returns:
        Dict[str, Any]: A dictionary with each (included) attribute name
        as key, and its serialized value as value.
    """

    _to_dict_exclude: Tuple[str, ...] = ()
    verbose: Optional[bool] = False

    def _serialize_callable(
        self, attr_value: Callable
    ) -> Dict[str, Any]:
        """
        Serialize a callable attribute (function, method, lambda, etc.) to a dict
        containing its name and docstring.

        Args:
            attr_value (Callable): The callable to serialize.

        Returns:
            dict: Dictionary with 'name' and 'doc' keys.
        """
        return {
            "name": getattr(
                attr_value, "__name__", type(attr_value).__name__
            ),
            "doc": getattr(attr_value, "__doc__", None),
        }

    def _serialize_attr(self, attr_name: str, attr_value: Any) -> Any:
        """
        Serialize a single attribute by attempting the following:
        - If it's callable, serialize its name and docstring.
        - If it exposes `to_dict`, recurse into that.
        - If it's JSON serializable, return as-is.
        - Otherwise, return a placeholder marking it as non-serializable.

        Args:
            attr_name (str): The name of the attribute.
            attr_value (Any): The value of the attribute.

        Returns:
            Any: A serializable representation of the attribute.
        """
        try:
            if callable(attr_value):
                return self._serialize_callable(attr_value)
            if hasattr(attr_value, "to_dict"):
                return attr_value.to_dict()
            # Test if JSON serializable
            json.dumps(attr_value)
            return attr_value
        except (TypeError, ValueError):
            return f"<Non-serializable: {type(attr_value).__name__}>"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the __dict__ of the current object, respecting
        the exclusion list. Each attribute is processed to maximize
        JSON compatibility and introspectability.

        Returns:
            Dict[str, Any]: The serialized dictionary representation.
        """
        excluded: Iterable[str] = set(self._to_dict_exclude)
        return {
            attr_name: self._serialize_attr(attr_name, attr_value)
            for attr_name, attr_value in self.__dict__.items()
            if attr_name not in excluded
        }

    def _log(self, level: str, message: str) -> None:
        """Log a message through the loguru logger when verbose is enabled."""
        if not getattr(self, "verbose", False):
            return

        logger.opt(depth=1).log(level.upper(), message)
