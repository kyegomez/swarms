from swarms.agents.tool_manager import _clip


def test_clip_under_or_equal_limit():
    """Verify values under or at the character limit are untouched."""
    assert _clip("short", 10) == "short"
    assert _clip("exact", 5) == "exact"
    assert _clip("", 5) == ""


def test_clip_over_limit_indicates_omitted_chars():
    """Verify clipped values append an omitted character count."""
    clipped = _clip("hello world", 5)
    assert clipped == "hello... (+6 chars)"
    assert _clip("abcdefghij", 3) == "abc... (+7 chars)"


def test_clip_handles_non_string_types():
    """Verify _clip coerces non-string values to text."""
    data = {"key": "very long dictionary value that exceeds limit"}
    clipped = _clip(data, 10)
    assert clipped.startswith("{'key': 'v... (")
    assert "chars)" in clipped
