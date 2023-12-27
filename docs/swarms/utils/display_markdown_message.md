# display_markdown_message

# Module Name: `display_markdown_message`

## Introduction

`display_markdown_message` is a useful utility function for creating visually-pleasing markdown messages within Python scripts. This function automatically manages multiline strings with lots of indentation and makes single-line messages with ">" tags easy to read, providing users with convenient and elegant logging or messaging capacity.

## Function Definition and Arguments

Function Definition:
```python
def display_markdown_message(message: str, color: str = "cyan"):
    ```
This function accepts two parameters:

|Parameter  |Type |Default Value |Description |
|---        |---  |---           |---         |
|message    |str  |None          |This is the message that is to be displayed. This should be a string. It can contain markdown syntax.|
|color      |str  |"cyan"        |This allows you to choose the color of the message. Default is "cyan". Accepts any valid color name.|

## Functionality and Usage

This utility function is used to display a markdown formatted message on the console. It accepts a message as a string and an optional color for the message. The function is ideal for generating stylized print outputs such as headers, status updates or pretty notifications.

By default, any text within the string which is enclosed within `>` tags or `---` is treated specially:

-  Lines encased in `>` tags are rendered as a blockquote in markdown.
-  Lines consisting of `---` are rendered as horizontal rules.

The function automatically strips off leading and trailing whitespaces from any line within the message, maintaining aesthetic consistency in your console output.

### Usage Examples

#### Basic Example

```python
display_markdown_message("> This is an important message", color="red")
```

Output:
```md
> **This is an important message**
```

This example will print out the string "This is an important message" in red color, enclosed in a blockquote tag.

#### Multiline Example

```python
message = """
> Header

My normal message here.

---

Another important information
"""
display_markdown_message(message, color="green")
```

Output:
```md
> **Header**

My normal message here.
_____

Another important information
```
The output is a green colored markdown styled text with the "Header" enclosed in a blockquote, followed by the phrase "My normal message here", a horizontal rule, and finally another phrase, "Another important information".

## Additional Information

Use newline characters `\n` to separate the lines of the message. Remember, each line of the message is stripped of leading and trailing whitespaces. If you have special markdown requirements, you may need to revise the input message string accordingly.

Also, keep in mind the console or terminal's ability to display the chosen color. If a particular console does not support the chosen color, the output may fallback to the default console color.

For a full list of color names supported by the `Console` module, refer to the official [Console documentation](http://console.readthedocs.io/).

## References and Resources

- Python Strings: https://docs.python.org/3/tutorial/introduction.html#strings
- Python Markdown: https://pypi.org/project/markdown/
- Console module: https://console.readthedocs.io/
