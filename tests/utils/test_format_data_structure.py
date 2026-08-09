import pytest
from swarms.utils.index import format_data_structure


class TestFormatDataStructure:
    """Table tests for format_data_structure with both style modes."""

    @pytest.mark.parametrize(
        "data,style,expected_lines",
        [
            (
                {"a": 1, "b": 2},
                "indented",
                ["a: 1", "b: 2"],
            ),
            (
                {"a": 1, "b": 2},
                "compact",
                ["a: 1", "b: 2"],
            ),
            (
                [1, 2, 3],
                "indented",
                ["1", "2", "3"],
            ),
            (
                [1, 2, 3],
                "compact",
                ["[1, 2, 3]"],
            ),
            (
                None,
                "compact",
                ["None"],
            ),
            (
                None,
                "indented",
                ["None"],
            ),
            (
                "hello",
                "compact",
                ['"hello"'],
            ),
            (
                "hello",
                "indented",
                ["hello"],
            ),
            (
                42,
                "compact",
                ["42"],
            ),
            (
                [],
                "compact",
                ["[]"],
            ),
            (
                [],
                "indented",
                ["[] (empty list)"],
            ),
            (
                {},
                "indented",
                ["{} (empty dict)"],
            ),
            (
                (True, False),
                "compact",
                ["(True, False)"],
            ),
            (
                {"user": {"id": 123, "active": True}, "data": [1, 2, 3]},
                "compact",
                ["user:", "id: 123", "active: True", "data: [1, 2, 3]"],
            ),
            (
                {"user": {"id": 123, "active": True}, "data": [1, 2, 3]},
                "indented",
                ["user:", "id: 123", "active: True", "data:", "1", "2", "3"],
            ),
        ],
    )
    def test_parametrized(self, data, style, expected_lines):
        result = format_data_structure(data, style=style)
        for line in expected_lines:
            assert line in result, f"Expected {line!r} in {result!r}"


class TestAnyToStrAlias:
    """Verify any_to_str produces the same output as compact style."""

    def test_any_to_str_matches_compact(self):
        from swarms.utils.any_to_str import any_to_str

        fixtures = [
            {"a": 1, "b": 2},
            [1, 2, 3],
            None,
            "hello",
            42,
            [],
            {},
            (True, False, None),
            [1, "text", None, 2.5],
            {"user": {"id": 123, "details": {"city": "New York"}}},
        ]
        for data in fixtures:
            a = any_to_str(data)
            b = format_data_structure(data, style="compact")
            assert a == b, f"Mismatch for {data!r}: {a!r} != {b!r}"
