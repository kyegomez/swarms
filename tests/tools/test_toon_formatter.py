"""
Tests for TOON Formatter

This test suite ensures the TOON formatter correctly encodes,
decodes, and compresses JSON data while maintaining data integrity.

Coverage Areas:
- Basic encode/decode operations
- Compression ratio calculations
- Edge cases and error handling
- Schema-aware operations
- Abbreviation system
"""

import json
import pytest
from swarms.utils.toon_formatter import (
    TOONFormatter,
    toon_encode,
    toon_decode,
    optimize_for_llm,
)


class TestTOONFormatterBasic:
    """Test basic TOON formatter operations."""

    def test_simple_encode(self):
        """Test encoding simple dictionary."""
        formatter = TOONFormatter()
        data = {"user": "Alice", "age": 30}

        toon_str = formatter.encode(data)

        assert isinstance(toon_str, str)
        assert "usr:Alice" in toon_str or "user:Alice" in toon_str
        assert "age:30" in toon_str

    def test_simple_decode(self):
        """Test decoding simple TOON string."""
        formatter = TOONFormatter(compact_keys=False)
        toon_str = "user:Alice age:30"

        decoded = formatter.decode(toon_str)

        assert decoded == {"user": "Alice", "age": 30}

    def test_roundtrip(self):
        """Test encode-decode roundtrip preserves data."""
        formatter = TOONFormatter(compact_keys=False)
        data = {
            "name": "Alice",
            "age": 30,
            "email": "alice@example.com",
            "active": True,
        }

        toon_str = formatter.encode(data)
        decoded = formatter.decode(toon_str)

        # Normalize boolean representation
        if "active" in decoded and decoded["active"] in [1, "1"]:
            decoded["active"] = True

        assert decoded == data

    def test_null_omission(self):
        """Test that null values are omitted when configured."""
        formatter = TOONFormatter(omit_null=True)
        data = {"name": "Alice", "age": None, "email": "alice@test.com"}

        toon_str = formatter.encode(data)

        # Should not contain the null age
        assert "age" not in toon_str
        assert "name" in toon_str or "nm" in toon_str

    def test_boolean_compression(self):
        """Test boolean compression to 1/0."""
        formatter = TOONFormatter()
        data = {"active": True, "verified": False}

        toon_str = formatter.encode(data)

        assert ":1" in toon_str  # True -> 1
        assert ":0" in toon_str  # False -> 0


class TestTOONFormatterAbbreviations:
    """Test key abbreviation system."""

    def test_common_abbreviations(self):
        """Test that common keys are abbreviated."""
        formatter = TOONFormatter(compact_keys=True)
        data = {
            "user": "Alice",
            "email": "alice@test.com",
            "status": "active",
        }

        toon_str = formatter.encode(data)

        # Check for abbreviated keys
        assert "usr:" in toon_str
        assert "eml:" in toon_str
        assert "sts:" in toon_str

    def test_reverse_abbreviations(self):
        """Test decoding abbreviated keys back to full names."""
        formatter = TOONFormatter(compact_keys=True)
        toon_str = "usr:Alice eml:alice@test.com sts:active"

        decoded = formatter.decode(toon_str)

        assert "user" in decoded
        assert "email" in decoded
        assert "status" in decoded

    def test_no_abbreviation_mode(self):
        """Test that compact_keys=False preserves original keys."""
        formatter = TOONFormatter(compact_keys=False)
        data = {"user": "Alice", "email": "alice@test.com"}

        toon_str = formatter.encode(data)

        assert "user:" in toon_str
        assert "email:" in toon_str
        assert "usr:" not in toon_str
        assert "eml:" not in toon_str


class TestTOONFormatterCompression:
    """Test compression metrics and calculations."""

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        formatter = TOONFormatter(compact_keys=True, omit_null=True)
        data = {
            "username": "Alice Johnson",
            "email": "alice@example.com",
            "status": "active",
            "created_at": "2025-01-15",
        }

        ratio = formatter.estimate_compression_ratio(data)

        # Should have meaningful compression
        assert 0.2 <= ratio <= 0.8
        assert isinstance(ratio, float)

    def test_compression_effectiveness(self):
        """Test that TOON is shorter than JSON."""
        formatter = TOONFormatter()
        data = {"user": "Alice", "age": 30, "email": "alice@test.com"}

        json_str = json.dumps(data)
        toon_str = formatter.encode(data)

        assert len(toon_str) < len(json_str)


class TestTOONFormatterEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dict(self):
        """Test encoding empty dictionary."""
        formatter = TOONFormatter()
        data = {}

        toon_str = formatter.encode(data)

        assert toon_str == ""

    def test_nested_dict(self):
        """Test encoding nested dictionary."""
        formatter = TOONFormatter()
        data = {
            "user": {"name": "Alice", "age": 30},
            "status": "active",
        }

        toon_str = formatter.encode(data)

        # Should contain nested structure
        assert "user:" in toon_str or "usr:" in toon_str
        assert "name:" in toon_str or "nm:" in toon_str

    def test_array_encoding(self):
        """Test encoding arrays."""
        formatter = TOONFormatter()
        data = {"users": ["Alice", "Bob", "Charlie"]}

        toon_str = formatter.encode(data)

        assert "[" in toon_str
        assert "]" in toon_str
        assert "Alice" in toon_str

    def test_special_characters(self):
        """Test handling of special characters."""
        formatter = TOONFormatter()
        data = {"name": "Alice:Smith", "description": "A test user"}

        toon_str = formatter.encode(data)

        # Should escape colons
        assert "Alice\\:Smith" in toon_str or "Alice:Smith" in toon_str

    def test_numeric_values(self):
        """Test encoding various numeric types."""
        formatter = TOONFormatter()
        data = {"int": 42, "float": 3.14, "negative": -10}

        toon_str = formatter.encode(data)

        assert "42" in toon_str
        assert "3.14" in toon_str
        assert "-10" in toon_str

    def test_max_depth_handling(self):
        """Test max depth limit for nested structures."""
        formatter = TOONFormatter(max_depth=2)

        # Create deeply nested structure
        data = {"a": {"b": {"c": {"d": "deep"}}}}

        # Should not raise error, may fall back to JSON
        toon_str = formatter.encode(data)
        assert isinstance(toon_str, str)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_toon_encode_function(self):
        """Test toon_encode convenience function."""
        data = {"user": "Alice", "age": 30}

        toon_str = toon_encode(data)

        assert isinstance(toon_str, str)
        assert "Alice" in toon_str

    def test_toon_decode_function(self):
        """Test toon_decode convenience function."""
        toon_str = "user:Alice age:30"

        data = toon_decode(toon_str)

        assert isinstance(data, dict)
        assert "user" in data or "age" in data

    def test_optimize_for_llm_toon(self):
        """Test optimize_for_llm with TOON format."""
        data = {"user": "Alice", "email": "alice@test.com"}

        optimized = optimize_for_llm(data, format="toon")

        assert isinstance(optimized, str)
        assert len(optimized) > 0

    def test_optimize_for_llm_json(self):
        """Test optimize_for_llm with JSON format."""
        data = {"user": "Alice", "age": 30}

        optimized = optimize_for_llm(data, format="json")

        assert isinstance(optimized, str)
        # Should be valid JSON
        parsed = json.loads(optimized)
        assert parsed == data

    def test_optimize_for_llm_compact(self):
        """Test optimize_for_llm with compact format."""
        data = {"user": "Alice", "age": 30}

        optimized = optimize_for_llm(data, format="compact")

        assert isinstance(optimized, str)
        # Should be compact (no spaces)
        assert " " not in optimized or optimized.count(" ") < 5


class TestTOONFormatterIntegration:
    """Test integration scenarios."""

    def test_large_dataset(self):
        """Test encoding large dataset."""
        formatter = TOONFormatter()

        # Create large dataset
        data = {
            "users": [
                {
                    "id": i,
                    "name": f"User{i}",
                    "email": f"user{i}@test.com",
                    "active": i % 2 == 0,
                }
                for i in range(100)
            ]
        }

        toon_str = formatter.encode(data)

        # Should compress significantly
        json_len = len(json.dumps(data))
        toon_len = len(toon_str)

        assert toon_len < json_len

    def test_schema_aware_encoding(self):
        """Test schema-aware encoding (basic)."""
        formatter = TOONFormatter()

        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
            },
        }

        data = {"id": 1, "name": "Alice"}

        # Should not raise error with schema
        toon_str = formatter.encode(data, schema=schema)
        assert isinstance(toon_str, str)


# Performance benchmarks (optional, can be run with pytest-benchmark)
class TestTOONFormatterPerformance:
    """Performance benchmarks for TOON formatter."""

    def test_encode_performance(self):
        """Test encoding performance."""
        formatter = TOONFormatter()
        data = {
            "users": [
                {"id": i, "name": f"User{i}", "active": True}
                for i in range(50)
            ]
        }

        import time

        start = time.time()
        for _ in range(10):
            formatter.encode(data)
        duration = time.time() - start

        # Should be reasonably fast (< 1 second for 10 iterations)
        assert duration < 1.0

    def test_decode_performance(self):
        """Test decoding performance."""
        formatter = TOONFormatter(compact_keys=False)
        toon_str = " ".join([f"id:{i} name:User{i} active:1" for i in range(50)])

        import time

        start = time.time()
        for _ in range(10):
            formatter.decode(toon_str)
        duration = time.time() - start

        # Should be reasonably fast
        assert duration < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
