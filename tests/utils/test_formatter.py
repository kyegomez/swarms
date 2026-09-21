from swarms.utils.formatter import Formatter


def test_formatter():
    """Test the formatter with various markdown content."""
    formatter = Formatter(md=True)

    # Test 1: Basic markdown with headers
    content1 = """# Main Title

This is a paragraph with **bold** text and *italic* text.

## Section 1
- Item 1
- Item 2
- Item 3

### Subsection
This is another paragraph with `inline code`.
"""

    formatter.print_panel(
        content1, title="Test 1: Basic Markdown", style="bold blue"
    )

    # Test 2: Code blocks with syntax highlighting
    content2 = """## Code Examples

Here's a Python example:

```python
def hello_world():
    '''A simple hello world function.'''
    print("Hello, World!")
    return True
```

And here's some JavaScript:

```javascript
function greet(name) {
    console.log(`Hello, ${name}!`);
}
```

Plain text code block:

```
This is just plain text
without any syntax highlighting
```
"""

    formatter.print_panel(
        content2, title="Test 2: Code Blocks", style="bold green"
    )

    # Test 3: Mixed content
    content3 = """## Mixed Content Test

This paragraph includes **various** formatting options:
- Lists with `code`
- Links [like this](https://example.com)
- And more...

```python
# Python code with comments
class Example:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}!"
```

### Table Example

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |
"""

    formatter.print_panel(
        content3, title="Test 3: Mixed Content", style="bold magenta"
    )

    # Test 4: Edge cases
    content4 = """This content starts without a header

It should still be formatted correctly.

```
No language specified
```

Single line content."""

    formatter.print_panel(
        content4, title="Test 4: Edge Cases", style="bold yellow"
    )

    # Test 5: Empty content
    formatter.print_panel(
        "", title="Test 5: Empty Content", style="bold red"
    )

    # Test 6: Using print_markdown method
    content6 = """# Direct Markdown Rendering

This uses the `print_markdown` method directly.

```python
# Syntax highlighted code
result = 42 * 2
print(f"The answer is {result}")
```
"""

    formatter.print_markdown(
        content6, title="Test 6: Direct Markdown", border_style="cyan"
    )


def _director_panel(**kwargs) -> str:
    """Render the director panel and return it as plain text."""
    formatter = Formatter(md=False)
    with formatter.console.capture() as capture:
        formatter.print_director_task_distribution(**kwargs)
    return capture.get()


def test_director_panel_shows_the_plan():
    output = _director_panel(
        director_name="Director",
        orders=[{"agent_name": "W1", "task": "Do the work"}],
        plan="Split the work in two, then combine.",
    )

    assert "Plan" in output
    assert "Split the work in two" in output


def test_director_panel_titles_itself_with_the_director_name():
    output = _director_panel(
        director_name="ResearchDirector",
        orders=[{"agent_name": "W1", "task": "Do the work"}],
    )

    assert "Director Name: ResearchDirector" in output


def test_director_panel_does_not_truncate_a_long_order():
    tail = "and this final clause must survive the render"
    task = (
        "Identify three practical benefits of multi-agent systems, "
        "each with one concrete production example, covering "
        "scalability, fault tolerance and specialization, " + tail
    )
    assert len(task) > 160, "task must exceed the old 160-char cap"

    output = _director_panel(
        director_name="Director",
        orders=[{"agent_name": "Researcher", "task": task}],
    )

    assert "..." not in output
    # rich wraps, so check the words rather than the whole string
    for word in tail.split():
        assert word in output


def test_director_panel_does_not_parse_markup_in_a_task():
    output = _director_panel(
        director_name="Director",
        orders=[
            {
                "agent_name": "W1",
                "task": "Focus on [scalability] first",
            }
        ],
    )

    assert "[scalability]" in output


if __name__ == "__main__":
    test_formatter()
