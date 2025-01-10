# Managing Prompts in Production

The `Prompt` class provides a comprehensive solution for managing prompts, including advanced features like version control, autosaving, and logging. This guide will walk you through how to effectively use this class in a production environment, focusing on its core features, use cases, and best practices.

## Table of Contents

1. **Getting Started**
    - Installation and Setup
    - Creating a New Prompt
2. **Managing Prompt Content**
    - Editing Prompts
    - Retrieving Prompt Content
3. **Version Control**
    - Tracking Edits and History
    - Rolling Back to Previous Versions
4. **Autosaving Prompts**
    - Enabling and Configuring Autosave
    - Manually Triggering Autosave
5. **Logging and Telemetry**
6. **Handling Errors**
7. **Extending the Prompt Class**
    - Customizing the Save Mechanism
    - Integrating with Databases

---

## 1. Getting Started

### Installation and Setup

Before diving into how to use the `Prompt` class, ensure that you have the required dependencies installed:

```bash
pip3 install -U swarms
```


### Creating a New Prompt

To create a new instance of a `Prompt`, simply initialize it with the required attributes such as `content`:

```python
from swarms import Prompt

prompt = Prompt(
    content="This is my first prompt!",
    name="My First Prompt",
    description="A simple example prompt."
)

print(prompt)
```

This creates a new prompt with the current timestamp and a unique identifier.

---

## 2. Managing Prompt Content

### Editing Prompts

Once you have initialized a prompt, you can edit its content using the `edit_prompt` method. Each time the content is edited, a new version is stored in the `edit_history`, and the `last_modified_at` timestamp is updated.

```python
new_content = "This is an updated version of my prompt."
prompt.edit_prompt(new_content)
```

**Note**: If the new content is identical to the current content, an error will be raised to prevent unnecessary edits:

```python
try:
    prompt.edit_prompt("This is my first prompt!")  # Same as initial content
except ValueError as e:
    print(e)  # Output: New content must be different from the current content.
```

### Retrieving Prompt Content

You can retrieve the current prompt content using the `get_prompt` method:

```python
current_content = prompt.get_prompt()
print(current_content)  # Output: This is an updated version of my prompt.
```

This method also logs telemetry data, which includes both system information and prompt metadata.

---

## 3. Version Control

### Tracking Edits and History

The `Prompt` class automatically tracks every change made to the prompt. This is stored in the `edit_history` attribute as a list of previous versions.

```python
print(prompt.edit_history)  # Output: ['This is my first prompt!', 'This is an updated version of my prompt.']
```

The number of edits is also tracked using the `edit_count` attribute:

```python
print(prompt.edit_count)  # Output: 2
```

### Rolling Back to Previous Versions

If you want to revert a prompt to a previous version, you can use the `rollback` method, passing the version index you want to revert to:

```python
prompt.rollback(0)
print(prompt.get_prompt())  # Output: This is my first prompt!
```

The rollback operation is thread-safe, and any rollback also triggers a telemetry log.

---

## 4. Autosaving Prompts

### Enabling and Configuring Autosave

To automatically save prompts to storage after every change, you can enable the `autosave` feature when initializing the prompt:

```python
prompt = Prompt(
    content="This is my first prompt!",
    autosave=True,
    autosave_folder="my_prompts"  # Specify the folder within WORKSPACE_DIR
)
```

This will ensure that every edit or rollback action triggers an autosave to the specified folder.

### Manually Triggering Autosave

You can also manually trigger an autosave by calling the `_autosave` method (which is a private method typically used internally):

```python
prompt._autosave()  # Manually triggers autosaving
```

Autosaves are stored as JSON files in the folder specified by `autosave_folder` under the workspace directory (`WORKSPACE_DIR` environment variable).

---

## 5. Logging and Telemetry

The `Prompt` class integrates with the `loguru` logging library to provide detailed logs for every major action, such as editing, rolling back, and saving. The `log_telemetry` method captures and logs system data, including prompt metadata, for each operation.

Here's an example of a log when editing a prompt:

```bash
2024-10-10 10:12:34.567 | INFO  | Editing prompt a7b8f9. Current content: 'This is my first prompt!'
2024-10-10 10:12:34.789 | DEBUG | Prompt a7b8f9 updated. Edit count: 1. New content: 'This is an updated version of my prompt.'
```

You can extend logging by integrating the `log_telemetry` method with your own telemetry systems or databases:

```python
prompt.log_telemetry()
```

---

## 6. Handling Errors

Error handling in the `Prompt` class is robust and prevents common mistakes, such as editing with identical content or rolling back to an invalid version. Here's a common scenario:

### Editing with Identical Content

```python
try:
    prompt.edit_prompt("This is an updated version of my prompt.")
except ValueError as e:
    print(e)  # Output: New content must be different from the current content.
```

### Invalid Rollback Version

```python
try:
    prompt.rollback(10)  # Invalid version index
except IndexError as e:
    print(e)  # Output: Invalid version number for rollback.
```

Always ensure that version numbers passed to `rollback` are within the valid range of existing versions.

---

## 7. Extending the Prompt Class

### Customizing the Save Mechanism

The `Prompt` class currently includes a placeholder for saving and loading prompts from persistent storage. You can override the `save_to_storage` and `load_from_storage` methods to integrate with databases, cloud storage, or other persistent layers.

Here's how you can implement the save functionality:

```python
def save_to_storage(self):
    # Example of saving to a database or cloud storage
    data = self.model_dump()
    save_to_database(data)  # Custom function to save data
```

Similarly, you can implement a `load_from_storage` function to load the prompt from a storage location using its unique identifier (`id`).


## Full Example code with all methods

```python
from swarms.prompts.prompt import Prompt

# Example 1: Initializing a Financial Report Prompt
financial_prompt = Prompt(
    content="Q1 2024 Earnings Report: Initial Draft", autosave=True
)

# Output the initial state of the prompt
print("\n--- Example 1: Initializing Prompt ---")
print(f"Prompt ID: {financial_prompt.id}")
print(f"Content: {financial_prompt.content}")
print(f"Created At: {financial_prompt.created_at}")
print(f"Edit Count: {financial_prompt.edit_count}")
print(f"History: {financial_prompt.edit_history}")


# Example 2: Editing a Financial Report Prompt
financial_prompt.edit_prompt(
    "Q1 2024 Earnings Report: Updated Revenue Figures"
)

# Output the updated state of the prompt
print("\n--- Example 2: Editing Prompt ---")
print(f"Content after edit: {financial_prompt.content}")
print(f"Edit Count: {financial_prompt.edit_count}")
print(f"History: {financial_prompt.edit_history}")


# Example 3: Rolling Back to a Previous Version
financial_prompt.edit_prompt("Q1 2024 Earnings Report: Final Version")
financial_prompt.rollback(
    1
)  # Roll back to the second version (index 1)

# Output the state after rollback
print("\n--- Example 3: Rolling Back ---")
print(f"Content after rollback: {financial_prompt.content}")
print(f"Edit Count: {financial_prompt.edit_count}")
print(f"History: {financial_prompt.edit_history}")


# Example 4: Handling Invalid Rollback
print("\n--- Example 4: Invalid Rollback ---")
try:
    financial_prompt.rollback(
        5
    )  # Attempt an invalid rollback (out of bounds)
except IndexError as e:
    print(f"Error: {e}")


# Example 5: Preventing Duplicate Edits
print("\n--- Example 5: Preventing Duplicate Edits ---")
try:
    financial_prompt.edit_prompt(
        "Q1 2024 Earnings Report: Updated Revenue Figures"
    )  # Duplicate content
except ValueError as e:
    print(f"Error: {e}")


# Example 6: Retrieving the Prompt Content as a String
print("\n--- Example 6: Retrieving Prompt as String ---")
current_content = financial_prompt.get_prompt()
print(f"Current Prompt Content: {current_content}")


# Example 7: Simulating Financial Report Changes Over Time
print("\n--- Example 7: Simulating Changes Over Time ---")
# Initialize a new prompt representing an initial financial report draft
financial_prompt = Prompt(
    content="Q2 2024 Earnings Report: Initial Draft"
)

# Simulate several updates over time
financial_prompt.edit_prompt(
    "Q2 2024 Earnings Report: Updated Forecasts"
)
financial_prompt.edit_prompt(
    "Q2 2024 Earnings Report: Revenue Adjustments"
)
financial_prompt.edit_prompt("Q2 2024 Earnings Report: Final Review")

# Display full history
print(f"Final Content: {financial_prompt.content}")
print(f"Edit Count: {financial_prompt.edit_count}")
print(f"Edit History: {financial_prompt.edit_history}")



```

---

## 8. Conclusion

This guide covered how to effectively use the `Prompt` class in production environments, including core features like editing, version control, autosaving, and logging. By following the best practices outlined here, you can ensure that your prompts are managed efficiently, with minimal overhead and maximum flexibility.

The `Prompt` class is designed with scalability and robustness in mind, making it a great choice for managing prompt content in multi-agent architectures or any application where dynamic prompt management is required. Feel free to extend the functionality to suit your needs, whether it's integrating with persistent storage or enhancing logging mechanisms.

By using this architecture, you'll be able to scale your system effortlessly while maintaining detailed version control and history of every interaction with your prompts.