from swarms.prompts.max_loop_prompt import generate_reasoning_prompt


def test_generate_reasoning_prompt_no_raw_placeholders():
    """Verify that generate_reasoning_prompt does not leak {current_loop} placeholders."""
    prompt = generate_reasoning_prompt(3)
    assert "{current_loop}" not in prompt
    assert "{current_loop + 1}" not in prompt
    assert "{{current_loop}}" not in prompt
    assert "3 maximum loops" in prompt
    assert "### End of Loop" in prompt
    assert "In the final loop (loop 3)" in prompt


def test_generate_reasoning_prompt_different_loop_counts():
    """Verify prompt formatting for varying loop bounds."""
    for count in (2, 5, 10):
        prompt = generate_reasoning_prompt(count)
        assert f"{count} maximum loops" in prompt
        assert f"In the final loop (loop {count})" in prompt
        assert "{current_loop}" not in prompt
