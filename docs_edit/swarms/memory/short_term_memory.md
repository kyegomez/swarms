# Short-Term Memory Module Documentation

## Introduction
The Short-Term Memory module is a component of the SWARMS framework designed for managing short-term and medium-term memory in a multi-agent system. This documentation provides a detailed explanation of the Short-Term Memory module, its purpose, functions, and usage.

### Purpose
The Short-Term Memory module serves the following purposes:
1. To store and manage messages in short-term memory.
2. To provide functions for retrieving, updating, and clearing memory.
3. To facilitate searching for specific terms within the memory.
4. To enable saving and loading memory data to/from a file.

### Class Definition
```python
class ShortTermMemory(BaseStructure):
    def __init__(
        self,
        return_str: bool = True,
        autosave: bool = True,
        *args,
        **kwargs,
    ):
    ...
```

#### Parameters
| Parameter           | Type     | Default Value | Description                                                                                                      |
|---------------------|----------|---------------|------------------------------------------------------------------------------------------------------------------|
| `return_str`        | bool     | True          | If True, returns memory as a string.                                                                            |
| `autosave`          | bool     | True          | If True, enables automatic saving of memory data to a file.                                                     |
| `*args`, `**kwargs` |          |               | Additional arguments and keyword arguments (not used in the constructor but allowed for flexibility).          |

### Functions

#### 1. `add`
```python
def add(self, role: str = None, message: str = None, *args, **kwargs):
```

- Adds a message to the short-term memory.
- Parameters:
  - `role` (str, optional): Role associated with the message.
  - `message` (str, optional): The message to be added.
- Returns: The added memory.

##### Example 1: Adding a Message to Short-Term Memory
```python
memory.add(role="Agent 1", message="Received task assignment.")
```

##### Example 2: Adding Multiple Messages to Short-Term Memory
```python
messages = [("Agent 1", "Received task assignment."), ("Agent 2", "Task completed.")]
for role, message in messages:
    memory.add(role=role, message=message)
```

#### 2. `get_short_term`
```python
def get_short_term(self):
```

- Retrieves the short-term memory.
- Returns: The contents of the short-term memory.

##### Example: Retrieving Short-Term Memory
```python
short_term_memory = memory.get_short_term()
for entry in short_term_memory:
    print(entry["role"], ":", entry["message"])
```

#### 3. `get_medium_term`
```python
def get_medium_term(self):
```

- Retrieves the medium-term memory.
- Returns: The contents of the medium-term memory.

##### Example: Retrieving Medium-Term Memory
```python
medium_term_memory = memory.get_medium_term()
for entry in medium_term_memory:
    print(entry["role"], ":", entry["message"])
```

#### 4. `clear_medium_term`
```python
def clear_medium_term(self):
```

- Clears the medium-term memory.

##### Example: Clearing Medium-Term Memory
```python
memory.clear_medium_term()
```

#### 5. `get_short_term_memory_str`
```python
def get_short_term_memory_str(self, *args, **kwargs):
```

- Retrieves the short-term memory as a string.
- Returns: A string representation of the short-term memory.

##### Example: Getting Short-Term Memory as a String
```python
short_term_memory_str = memory.get_short_term_memory_str()
print(short_term_memory_str)
```

#### 6. `update_short_term`
```python
def update_short_term(self, index, role: str, message: str, *args, **kwargs):
```

- Updates a message in the short-term memory.
- Parameters:
  - `index` (int): The index of the message to update.
  - `role` (str): New role for the message.
  - `message` (str): New message content.
- Returns: None.

##### Example: Updating a Message in Short-Term Memory
```python
memory.update_short_term(
    index=0, role="Updated Role", message="Updated message content."
)
```

#### 7. `clear`
```python
def clear(self):
```

- Clears the short-term memory.

##### Example: Clearing Short-Term Memory
```python
memory.clear()
```

#### 8. `search_memory`
```python
def search_memory(self, term):
```

- Searches the memory for a specific term.
- Parameters:
  - `term` (str): The term to search for.
- Returns: A dictionary containing search results for short-term and medium-term memory.

##### Example: Searching Memory for a Term
```python
search_results = memory.search_memory("task")
print("Short-Term Memory Results:", search_results["short_term"])
print("Medium-Term Memory Results:", search_results["medium_term"])
```

#### 9. `return_shortmemory_as_str`
```python
def return_shortmemory_as_str(self):
```

- Returns the memory as a string.

##### Example: Returning Short-Term Memory as a String
```python
short_term_memory_str = memory.return_shortmemory_as_str()
print(short_term_memory_str)
```

#### 10. `move_to_medium_term`
```python
def move_to_medium_term(self, index):
```

- Moves a message from the short-term memory to the medium-term memory.
- Parameters:
  - `index` (int): The index of the message to move.

##### Example: Moving a Message to Medium-Term Memory
```python
memory.move_to_medium_term(index=0)
```

#### 11. `return_medium_memory_as_str`
```python
def return_medium_memory_as_str(self):
```

- Returns the medium-term memory as a string.

##### Example: Returning Medium-Term Memory as a String
```python
medium_term_memory_str = memory.return_medium_memory_as_str()
print(medium_term_memory_str)
```

#### 12. `save_to_file`
```python
def save_to_file(self, filename: str):
```

- Saves the memory data to a file.
- Parameters:
  - `filename` (str): The name of the file to save the data to.

##### Example: Saving Memory Data to a File
```python
memory.save_to_file("memory_data.json")
```

#### 13. `load_from_file`
```python
def load_from_file(self, filename: str, *args, **kwargs):
```

- Loads memory data from a file.
- Parameters:
  - `filename` (str): The name of the file to load data from.

##### Example: Loading Memory Data from a File
```python
memory.load_from_file("memory_data.json")
```

### Additional Information and Tips

- To use the Short-Term Memory module effectively, consider the following tips:
  - Use the `add` function to store messages in short-term memory.
  -

 Retrieve memory contents using `get_short_term` and `get_medium_term` functions.
  - Clear memory as needed using `clear` and `clear_medium_term` functions.
  - Search for specific terms within the memory using the `search_memory` function.
  - Save and load memory data to/from files using `save_to_file` and `load_from_file` functions.

- Ensure proper exception handling when using memory functions to handle potential errors gracefully.

- When using the `search_memory` function, iterate through the results dictionary to access search results for short-term and medium-term memory.

### References and Resources

- For more information on multi-agent systems and memory management, refer to the SWARMS framework documentation: [SWARMS Documentation](https://swarms.apac.ai/).

- For advanced memory management and customization, explore the SWARMS framework source code.

