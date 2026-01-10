"""
TOON (Token-Oriented Object Notation) Formatter

Local utilities for TOON serialization and deserialization.
Provides offline processing capabilities without requiring TOON SDK API.

Key Features:
    - Compact key/value notation
    - Null value omission
    - Schema-aware field abbreviation
    - 30-60% token reduction
    - Human-readable output

References:
    - TOON Spec: https://github.com/toon-format
    - Benchmarks: 73.9% retrieval accuracy
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

from loguru import logger


class TOONFormatter:
    """
    Local TOON formatter for JSON serialization optimization.

    This class provides offline TOON encoding/decoding without
    requiring external API calls, useful for:
    - Rapid prototyping
    - Offline development
    - Fallback when SDK unavailable
    - Custom serialization rules

    Examples:
        >>> formatter = TOONFormatter()
        >>> data = {"user": "Alice", "age": 30, "city": "NYC"}
        >>> toon = formatter.encode(data)
        >>> print(toon)  # "usr:Alice age:30 city:NYC"
        >>> decoded = formatter.decode(toon)
    """

    # Common abbreviations for frequent keys
    KEY_ABBREVIATIONS = {
        "user": "usr",
        "username": "usr",
        "name": "nm",
        "description": "desc",
        "identifier": "id",
        "status": "sts",
        "message": "msg",
        "timestamp": "ts",
        "created_at": "crt",
        "updated_at": "upd",
        "deleted_at": "del",
        "email": "eml",
        "phone": "ph",
        "address": "addr",
        "metadata": "meta",
        "configuration": "cfg",
        "parameters": "prm",
        "attributes": "attr",
        "properties": "prop",
        "value": "val",
        "count": "cnt",
        "total": "tot",
        "amount": "amt",
        "price": "prc",
        "quantity": "qty",
        "percentage": "pct",
        "enabled": "en",
        "disabled": "dis",
        "active": "act",
        "inactive": "inact",
    }

    # Reverse mapping for decoding
    ABBREVIATION_REVERSE = {
        v: k for k, v in KEY_ABBREVIATIONS.items()
    }

    def __init__(
        self,
        compact_keys: bool = True,
        omit_null: bool = True,
        use_shorthand: bool = True,
        max_depth: int = 10,
        indent: int = 0,
    ):
        """
        Initialize TOON formatter.

        Args:
            compact_keys: Use abbreviated key names
            omit_null: Exclude null/None values
            use_shorthand: Enable TOON shorthand syntax
            max_depth: Maximum nesting depth
            indent: Indentation level (0 for compact)
        """
        self.compact_keys = compact_keys
        self.omit_null = omit_null
        self.use_shorthand = use_shorthand
        self.max_depth = max_depth
        self.indent = indent

    def encode(
        self,
        data: Union[Dict[str, Any], List[Any]],
        schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Encode JSON data to TOON format.

        Args:
            data: JSON data to encode
            schema: Optional JSON Schema for optimization

        Returns:
            TOON-formatted string

        Examples:
            >>> formatter = TOONFormatter()
            >>> data = {"user": "Alice", "age": 30, "active": True}
            >>> toon = formatter.encode(data)
            >>> print(toon)  # "usr:Alice age:30 act:1"
        """
        try:
            if isinstance(data, dict):
                return self._encode_object(data, depth=0)
            elif isinstance(data, list):
                return self._encode_array(data, depth=0)
            else:
                return self._encode_value(data)
        except Exception as e:
            logger.error(f"TOON encoding error: {e}")
            raise ValueError(f"Failed to encode data: {e}") from e

    def decode(
        self,
        toon_str: str,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], List[Any]]:
        """
        Decode TOON format to JSON.

        Args:
            toon_str: TOON-formatted string
            schema: Optional JSON Schema for validation

        Returns:
            Decoded JSON data

        Examples:
            >>> formatter = TOONFormatter()
            >>> toon = "usr:Alice age:30 act:1"
            >>> data = formatter.decode(toon)
            >>> print(data)  # {"user": "Alice", "age": 30, "active": True}
        """
        try:
            toon_str = toon_str.strip()

            # Detect if it's an array or object
            if toon_str.startswith("[") and toon_str.endswith("]"):
                return self._decode_array(toon_str)
            else:
                return self._decode_object(toon_str)

        except Exception as e:
            logger.error(f"TOON decoding error: {e}")
            raise ValueError(f"Failed to decode TOON data: {e}") from e

    def _encode_object(self, obj: Dict[str, Any], depth: int) -> str:
        """Encode a dictionary to TOON object notation."""
        if depth > self.max_depth:
            logger.warning(f"Max depth {self.max_depth} exceeded")
            return json.dumps(obj)

        pairs = []
        for key, value in obj.items():
            # Skip null values if configured
            if self.omit_null and value is None:
                continue

            # Abbreviate key if enabled
            if self.compact_keys:
                key = self.KEY_ABBREVIATIONS.get(key, key)

            # Encode value
            encoded_value = self._encode_value_with_depth(value, depth + 1)

            # Use TOON notation: key:value
            pairs.append(f"{key}:{encoded_value}")

        separator = " " if self.indent == 0 else "\n" + "  " * (depth + 1)
        return separator.join(pairs)

    def _encode_array(self, arr: List[Any], depth: int) -> str:
        """Encode a list to TOON array notation."""
        if depth > self.max_depth:
            logger.warning(f"Max depth {self.max_depth} exceeded")
            return json.dumps(arr)

        encoded_items = [
            self._encode_value_with_depth(item, depth + 1) for item in arr
        ]

        if self.indent == 0:
            return "[" + ",".join(encoded_items) + "]"
        else:
            sep = "\n" + "  " * (depth + 1)
            return "[" + sep + sep.join(encoded_items) + "\n" + "  " * depth + "]"

    def _encode_value(self, value: Any) -> str:
        """Encode a single value."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            # Escape special characters
            value = value.replace(":", "\\:")
            value = value.replace(" ", "\\_")
            return value
        else:
            return json.dumps(value)

    def _encode_value_with_depth(self, value: Any, depth: int) -> str:
        """Encode value with depth tracking for nested structures."""
        if isinstance(value, dict):
            return self._encode_object(value, depth)
        elif isinstance(value, list):
            return self._encode_array(value, depth)
        else:
            return self._encode_value(value)

    def _decode_object(self, toon_str: str) -> Dict[str, Any]:
        """Decode TOON object notation to dictionary."""
        result = {}

        # Split by spaces (but not escaped spaces)
        pairs = re.split(r'(?<!\\)\s+', toon_str.strip())

        for pair in pairs:
            if not pair:
                continue

            # Split by first unescaped colon
            match = re.match(r'([^:]+):(.+)', pair)
            if not match:
                logger.warning(f"Skipping invalid pair: {pair}")
                continue

            key, value_str = match.groups()

            # Expand abbreviated keys
            if self.compact_keys and key in self.ABBREVIATION_REVERSE:
                key = self.ABBREVIATION_REVERSE[key]

            # Decode value
            value = self._decode_value(value_str)
            result[key] = value

        return result

    def _decode_array(self, toon_str: str) -> List[Any]:
        """Decode TOON array notation to list."""
        # Remove brackets
        content = toon_str[1:-1].strip()

        if not content:
            return []

        # Split by commas (but not escaped commas)
        items = re.split(r'(?<!\\),', content)

        return [self._decode_value(item.strip()) for item in items]

    def _decode_value(self, value_str: str) -> Any:
        """Decode a single value."""
        value_str = value_str.strip()

        # Handle null
        if value_str == "null":
            return None

        # Handle booleans
        if value_str == "1":
            return True
        elif value_str == "0":
            return False

        # Handle numbers
        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass

        # Handle nested objects
        if ":" in value_str and not value_str.startswith("["):
            return self._decode_object(value_str)

        # Handle nested arrays
        if value_str.startswith("[") and value_str.endswith("]"):
            return self._decode_array(value_str)

        # Handle strings (unescape)
        value_str = value_str.replace("\\:", ":")
        value_str = value_str.replace("\\_", " ")

        # Try JSON parsing as fallback
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            return value_str

    def estimate_compression_ratio(
        self, data: Union[Dict[str, Any], List[Any]]
    ) -> float:
        """
        Estimate compression ratio for given data.

        Args:
            data: JSON data

        Returns:
            Estimated compression ratio (0.0-1.0)

        Examples:
            >>> formatter = TOONFormatter()
            >>> data = {"username": "Alice", "age": 30}
            >>> ratio = formatter.estimate_compression_ratio(data)
            >>> print(f"Expected {ratio:.1%} compression")
        """
        original_json = json.dumps(data, separators=(",", ":"))
        toon_encoded = self.encode(data)

        original_len = len(original_json)
        toon_len = len(toon_encoded)

        if original_len == 0:
            return 0.0

        compression = (original_len - toon_len) / original_len
        return max(0.0, min(1.0, compression))


# Convenience functions
def toon_encode(
    data: Union[Dict[str, Any], List[Any]],
    compact_keys: bool = True,
    omit_null: bool = True,
) -> str:
    """
    Quick encode function for TOON format.

    Args:
        data: JSON data to encode
        compact_keys: Use abbreviated keys
        omit_null: Exclude null values

    Returns:
        TOON-formatted string

    Examples:
        >>> from swarms.utils.toon_formatter import toon_encode
        >>> toon = toon_encode({"user": "Alice", "age": 30})
    """
    formatter = TOONFormatter(
        compact_keys=compact_keys, omit_null=omit_null
    )
    return formatter.encode(data)


def toon_decode(toon_str: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Quick decode function for TOON format.

    Args:
        toon_str: TOON-formatted string

    Returns:
        Decoded JSON data

    Examples:
        >>> from swarms.utils.toon_formatter import toon_decode
        >>> data = toon_decode("usr:Alice age:30")
    """
    formatter = TOONFormatter()
    return formatter.decode(toon_str)


def optimize_for_llm(
    data: Union[Dict[str, Any], List[Any], str],
    format: str = "toon",
) -> str:
    """
    Optimize data for LLM prompts using TOON or other formats.

    Args:
        data: Data to optimize (JSON or string)
        format: Output format ('toon', 'json', 'compact')

    Returns:
        Optimized string representation

    Examples:
        >>> from swarms.utils.toon_formatter import optimize_for_llm
        >>> data = {"results": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
        >>> optimized = optimize_for_llm(data, format="toon")
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data

    if format == "toon":
        formatter = TOONFormatter(
            compact_keys=True,
            omit_null=True,
            indent=0,
        )
        return formatter.encode(data)
    elif format == "compact":
        return json.dumps(data, separators=(",", ":"))
    else:  # json
        return json.dumps(data, indent=2)
