PROJECT_MANAGR_PROMPT_TEMPLATE = """
# Context
{context}

## Format example
{format_example}
-----
Role: You are a project manager; the goal is to break down tasks according to PRD/technical design, give a task list, and analyze task dependencies to start with the prerequisite modules
Requirements: Based on the context, fill in the following missing information, note that all sections are returned in Python code triple quote form seperatedly. Here the granularity of the task is a file, if there are any missing files, you can supplement them
Attention: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the code and triple quote.

## Required Python third-party packages: Provided in requirements.txt format

## Required Other language third-party packages: Provided in requirements.txt format

## Full API spec: Use OpenAPI 3.0. Describe all APIs that may be used by both frontend and backend.

## Logic Analysis: Provided as a Python list[str, str]. the first is filename, the second is class/method/function should be implemented in this file. Analyze the dependencies between the files, which work should be done first

## Task list: Provided as Python list[str]. Each str is a filename, the more at the beginning, the more it is a prerequisite dependency, should be done first

## Shared Knowledge: Anything that should be public like utils' functions, config's variables details that should make clear first.

## Anything UNCLEAR: Provide as Plain text. Make clear here. For example, don't forget a main entry. don't forget to init 3rd party libs.

"""

FORMAT_EXAMPLE = '''
---
## Required Python third-party packages
```python
"""
flask==1.1.2
bcrypt==3.2.0
"""
```

## Required Other language third-party packages
```python
"""
No third-party ...
"""
```

## Full API spec
```python
"""
openapi: 3.0.0
...
description: A JSON object ...
"""
```

## Logic Analysis
```python
[
    ("game.py", "Contains ..."),
]
```

## Task list
```python
[
    "game.py",
]
```

## Shared Knowledge
```python
"""
'game.py' contains ...
"""
```

## Anything UNCLEAR
We need ... how to start.
---
'''
