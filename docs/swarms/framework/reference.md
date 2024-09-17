# API Reference Documentation



### `swarms.__init__`

**Description**:  
This module initializes the Swarms package by concurrently executing the bootup process and activating Sentry for telemetry. It imports various components from other modules within the Swarms package.

**Imports**:
- `concurrent.futures`: A module that provides a high-level interface for asynchronously executing callables.
- `swarms.telemetry.bootup`: Contains the `bootup` function for initializing telemetry.
- `swarms.telemetry.sentry_active`: Contains the `activate_sentry` function to enable Sentry for error tracking.
- Other modules from the Swarms package are imported for use, including agents, artifacts, prompts, structs, telemetry, tools, utils, and schemas.

**Concurrent Execution**:
The module uses `ThreadPoolExecutor` to run the `bootup` and `activate_sentry` functions concurrently.

```python
import concurrent.futures
from swarms.telemetry.bootup import bootup  # noqa: E402, F403
from swarms.telemetry.sentry_active import activate_sentry

# Use ThreadPoolExecutor to run bootup and activate_sentry concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(bootup)
    executor.submit(activate_sentry)

from swarms.agents import *  # noqa: E402, F403
from swarms.artifacts import *  # noqa: E402, F403
from swarms.prompts import *  # noqa: E402, F403
from swarms.structs import *  # noqa: E402, F403
from swarms.telemetry import *  # noqa: E402, F403
from swarms.tools import *  # noqa: E402, F403
from swarms.utils import *  # noqa: E402, F403
from swarms.schemas import *  # noqa: E402, F403
```

**Note**: There are no documentable functions or classes within this module itself, as it primarily serves to execute initial setup tasks and import other modules.




### `swarms.artifacts.base_artifact`

**Description**:  
This module defines the `BaseArtifact` abstract base class for representing artifacts in the system. It provides methods to convert artifact values to various formats and enforces the implementation of an addition method for subclasses.

**Imports**:
- `json`: A module for parsing JSON data.
- `uuid`: A module for generating unique identifiers.
- `ABC`, `abstractmethod`: Tools from the `abc` module to define abstract base classes.
- `dataclass`: A decorator for creating data classes.
- `Any`: A type hint for any data type.

### `BaseArtifact`
**Description**:  
An abstract base class for artifacts that includes common attributes and methods for handling artifact values.

**Attributes**:  
- `id` (`str`): A unique identifier for the artifact, generated if not provided.
- `name` (`str`): The name of the artifact. If not provided, it defaults to the artifact's ID.
- `value` (`Any`): The value associated with the artifact.

**Methods**:

- `__post_init__(self) -> None`
    - **Description**: Initializes the artifact, setting the `id` and `name` attributes if they are not provided.
    - **Parameters**: None.
    - **Return**: None.

- `value_to_bytes(cls, value: Any) -> bytes`
    - **Description**: Converts the given value to bytes.
    - **Parameters**: 
      - `value` (`Any`): The value to convert.
    - **Return**: 
      - (`bytes`): The value converted to bytes.

- `value_to_dict(cls, value: Any) -> dict`
    - **Description**: Converts the given value to a dictionary.
    - **Parameters**: 
      - `value` (`Any`): The value to convert.
    - **Return**: 
      - (`dict`): The value converted to a dictionary.

- `to_text(self) -> str`
    - **Description**: Converts the artifact's value to a text representation.
    - **Parameters**: None.
    - **Return**: 
      - (`str`): The string representation of the artifact's value.

- `__str__(self) -> str`
    - **Description**: Returns a string representation of the artifact.
    - **Parameters**: None.
    - **Return**: 
      - (`str`): The string representation of the artifact.

- `__bool__(self) -> bool`
    - **Description**: Returns the boolean value of the artifact based on its value.
    - **Parameters**: None.
    - **Return**: 
      - (`bool`): The boolean value of the artifact.

- `__len__(self) -> int`
    - **Description**: Returns the length of the artifact's value.
    - **Parameters**: None.
    - **Return**: 
      - (`int`): The length of the artifact's value.

- `__add__(self, other: BaseArtifact) -> BaseArtifact`
    - **Description**: Abstract method for adding two artifacts together. Must be implemented by subclasses.
    - **Parameters**: 
      - `other` (`BaseArtifact`): The other artifact to add.
    - **Return**: 
      - (`BaseArtifact`): The result of adding the two artifacts.

**Example**:
```python
from swarms.artifacts.base_artifact import BaseArtifact

class MyArtifact(BaseArtifact):
    def __add__(self, other: BaseArtifact) -> BaseArtifact:
        return MyArtifact(id=self.id, name=self.name, value=self.value + other.value)

artifact1 = MyArtifact(id="123", name="Artifact1", value=10)
artifact2 = MyArtifact(id="456", name="Artifact2", value=20)
result = artifact1 + artifact2
print(result)  # Output: MyArtifact with the combined value
```




### `swarms.artifacts.text_artifact`

**Description**:  
This module defines the `TextArtifact` class, which represents a text-based artifact. It extends the `BaseArtifact` class and includes attributes and methods specific to handling text values, including encoding options, embedding generation, and token counting.

**Imports**:
- `dataclass`, `field`: Decorators and functions from the `dataclasses` module for creating data classes.
- `Callable`: A type hint indicating a callable object from the `typing` module.
- `BaseArtifact`: The abstract base class for artifacts, imported from `swarms.artifacts.base_artifact`.

### `TextArtifact`
**Description**:  
Represents a text artifact with additional functionality for handling text values, encoding, and embeddings.

**Attributes**:  
- `value` (`str`): The text value of the artifact.
- `encoding` (`str`, optional): The encoding of the text (default is "utf-8").
- `encoding_error_handler` (`str`, optional): The error handler for encoding errors (default is "strict").
- `tokenizer` (`Callable`, optional): A callable for tokenizing the text value.
- `_embedding` (`list[float]`): The embedding of the text artifact (default is an empty list).

**Properties**:
- `embedding` (`Optional[list[float]]`): Returns the embedding of the text artifact if available; otherwise, returns `None`.

**Methods**:

- `__add__(self, other: BaseArtifact) -> TextArtifact`
    - **Description**: Concatenates the text value of this artifact with the text value of another artifact.
    - **Parameters**: 
      - `other` (`BaseArtifact`): The other artifact to concatenate with.
    - **Return**: 
      - (`TextArtifact`): A new `TextArtifact` instance with the concatenated value.

- `__bool__(self) -> bool`
    - **Description**: Checks if the text value of the artifact is non-empty.
    - **Parameters**: None.
    - **Return**: 
      - (`bool`): `True` if the text value is non-empty; otherwise, `False`.

- `generate_embedding(self, model) -> list[float] | None`
    - **Description**: Generates the embedding of the text artifact using a given embedding model.
    - **Parameters**: 
      - `model`: An embedding model that provides the `embed_string` method.
    - **Return**: 
      - (`list[float] | None`): The generated embedding as a list of floats, or `None` if the embedding could not be generated.

- `token_count(self) -> int`
    - **Description**: Counts the number of tokens in the text artifact using a specified tokenizer.
    - **Parameters**: None.
    - **Return**: 
      - (`int`): The number of tokens in the text value.

- `to_bytes(self) -> bytes`
    - **Description**: Converts the text value of the artifact to bytes using the specified encoding and error handler.
    - **Parameters**: None.
    - **Return**: 
      - (`bytes`): The text value encoded as bytes.

**Example**:
```python
from swarms.artifacts.text_artifact import TextArtifact

# Create a TextArtifact instance
text_artifact = TextArtifact(value="Hello, World!")

# Generate embedding (assuming an appropriate model is provided)
# embedding = text_artifact.generate_embedding(model)

# Count tokens in the text artifact
token_count = text_artifact.token_count()

# Convert to bytes
bytes_value = text_artifact.to_bytes()

print(text_artifact)  # Output: Hello, World!
print(token_count)    # Output: Number of tokens
print(bytes_value)    # Output: b'Hello, World!'
```




### `swarms.artifacts.main_artifact`

**Description**:  
This module defines the `Artifact` class, which represents a file artifact with versioning capabilities. It allows for the creation, editing, saving, loading, and exporting of file artifacts, as well as managing their version history. The module also includes a `FileVersion` class to encapsulate the details of each version of the artifact.

**Imports**:
- `time`: A module for time-related functions.
- `logger`: A logging utility from `swarms.utils.loguru_logger`.
- `os`: A module providing a way of using operating system-dependent functionality.
- `json`: A module for parsing JSON data.
- `List`, `Union`, `Dict`, `Any`: Type hints from the `typing` module.
- `BaseModel`, `Field`, `validator`: Tools from the `pydantic` module for data validation and settings management.
- `datetime`: A module for manipulating dates and times.

### `FileVersion`
**Description**:  
Represents a version of a file with its content and timestamp.

**Attributes**:  
- `version_number` (`int`): The version number of the file.
- `content` (`str`): The content of the file version.
- `timestamp` (`str`): The timestamp of the file version, formatted as "YYYY-MM-DD HH:MM:SS".

**Methods**:

- `__str__(self) -> str`
    - **Description**: Returns a string representation of the file version.
    - **Parameters**: None.
    - **Return**: 
      - (`str`): A formatted string containing the version number, timestamp, and content.

### `Artifact`
**Description**:  
Represents a file artifact with attributes to manage its content and version history.

**Attributes**:  
- `file_path` (`str`): The path to the file.
- `file_type` (`str`): The type of the file (e.g., ".txt").
- `contents` (`str`): The contents of the file.
- `versions` (`List[FileVersion]`): The list of file versions.
- `edit_count` (`int`): The number of times the file has been edited.

**Methods**:

- `validate_file_type(cls, v, values) -> str`
    - **Description**: Validates the file type based on the file extension.
    - **Parameters**: 
      - `v` (`str`): The file type to validate.
      - `values` (`dict`): A dictionary of other field values.
    - **Return**: 
      - (`str`): The validated file type.

- `create(self, initial_content: str) -> None`
    - **Description**: Creates a new file artifact with the initial content.
    - **Parameters**: 
      - `initial_content` (`str`): The initial content to set for the artifact.
    - **Return**: None.

- `edit(self, new_content: str) -> None`
    - **Description**: Edits the artifact's content, tracking the change in the version history.
    - **Parameters**: 
      - `new_content` (`str`): The new content to set for the artifact.
    - **Return**: None.

- `save(self) -> None`
    - **Description**: Saves the current artifact's contents to the specified file path.
    - **Parameters**: None.
    - **Return**: None.

- `load(self) -> None`
    - **Description**: Loads the file contents from the specified file path into the artifact.
    - **Parameters**: None.
    - **Return**: None.

- `get_version(self, version_number: int) -> Union[FileVersion, None]`
    - **Description**: Retrieves a specific version of the artifact by its version number.
    - **Parameters**: 
      - `version_number` (`int`): The version number to retrieve.
    - **Return**: 
      - (`FileVersion | None`): The requested version if found; otherwise, `None`.

- `get_contents(self) -> str`
    - **Description**: Returns the current contents of the artifact as a string.
    - **Parameters**: None.
    - **Return**: 
      - (`str`): The current contents of the artifact.

- `get_version_history(self) -> str`
    - **Description**: Returns the version history of the artifact as a formatted string.
    - **Parameters**: None.
    - **Return**: 
      - (`str`): A formatted string containing the version history.

- `export_to_json(self, file_path: str) -> None`
    - **Description**: Exports the artifact to a JSON file.
    - **Parameters**: 
      - `file_path` (`str`): The path to the JSON file where the artifact will be saved.
    - **Return**: None.

- `import_from_json(cls, file_path: str) -> "Artifact"`
    - **Description**: Imports an artifact from a JSON file.
    - **Parameters**: 
      - `file_path` (`str`): The path to the JSON file to import the artifact from.
    - **Return**: 
      - (`Artifact`): The imported artifact instance.

- `get_metrics(self) -> str`
    - **Description**: Returns all metrics of the artifact as a formatted string.
    - **Parameters**: None.
    - **Return**: 
      - (`str`): A string containing all metrics of the artifact.

- `to_dict(self) -> Dict[str, Any]`
    - **Description**: Converts the artifact instance to a dictionary representation.
    - **Parameters**: None.
    - **Return**: 
      - (`Dict[str, Any]`): The dictionary representation of the artifact.

- `from_dict(cls, data: Dict[str, Any]) -> "Artifact"`
    - **Description**: Creates an artifact instance from a dictionary representation.
    - **Parameters**: 
      - `data` (`Dict[str, Any]`): The dictionary to create the artifact from.
    - **Return**: 
      - (`Artifact`): The created artifact instance.

**Example**:
```python
from swarms.artifacts.main_artifact import Artifact

# Create an Artifact instance
artifact = Artifact(file_path="example.txt", file_type=".txt")
artifact.create("Initial content")
artifact.edit("First edit")
artifact.edit("Second edit")
artifact.save()

# Export to JSON
artifact.export_to_json("artifact.json")

# Import from JSON
imported_artifact = Artifact.import_from_json("artifact.json")

# Get metrics
print(artifact.get_metrics())
```




### `swarms.artifacts.__init__`

**Description**:  
This module serves as the initialization point for the artifacts subpackage within the Swarms framework. It imports and exposes the key classes related to artifacts, including `BaseArtifact`, `TextArtifact`, and `Artifact`, making them available for use in other parts of the application.

**Imports**:
- `BaseArtifact`: The abstract base class for artifacts, imported from `swarms.artifacts.base_artifact`.
- `TextArtifact`: A class representing text-based artifacts, imported from `swarms.artifacts.text_artifact`.
- `Artifact`: A class representing file artifacts with versioning capabilities, imported from `swarms.artifacts.main_artifact`.

**Exported Classes**:
- `BaseArtifact`: The base class for all artifacts.
- `TextArtifact`: A specialized artifact class for handling text values.
- `Artifact`: A class for managing file artifacts, including their content and version history.

**Example**:
```python
from swarms.artifacts import *

# Create instances of the artifact classes
base_artifact = BaseArtifact(id="1", name="Base Artifact", value="Some value")  # This will raise an error since BaseArtifact is abstract
text_artifact = TextArtifact(value="Sample text")
file_artifact = Artifact(file_path="example.txt", file_type=".txt")

# Use the classes as needed
print(text_artifact)  # Output: Sample text
``` 

**Note**: Since `BaseArtifact` is an abstract class, it cannot be instantiated directly.


# Agents

### `swarms.agents.__init__`

**Description**:  
This module serves as the initialization point for the agents subpackage within the Swarms framework. It imports and exposes key classes and functions related to agent operations, including stopping conditions and the `ToolAgent` class, making them available for use in other parts of the application.

**Imports**:
- `check_cancelled`: A function to check if the operation has been cancelled.
- `check_complete`: A function to check if the operation is complete.
- `check_done`: A function to check if the operation is done.
- `check_end`: A function to check if the operation has ended.
- `check_error`: A function to check if there was an error during the operation.
- `check_exit`: A function to check if the operation has exited.
- `check_failure`: A function to check if the operation has failed.
- `check_finished`: A function to check if the operation has finished.
- `check_stopped`: A function to check if the operation has been stopped.
- `check_success`: A function to check if the operation was successful.
- `ToolAgent`: A class representing an agent that utilizes tools.

**Exported Classes and Functions**:
- `ToolAgent`: The class for managing tool-based agents.
- `check_done`: Checks if the operation is done.
- `check_finished`: Checks if the operation has finished.
- `check_complete`: Checks if the operation is complete.
- `check_success`: Checks if the operation was successful.
- `check_failure`: Checks if the operation has failed.
- `check_error`: Checks if there was an error during the operation.
- `check_stopped`: Checks if the operation has been stopped.
- `check_cancelled`: Checks if the operation has been cancelled.
- `check_exit`: Checks if the operation has exited.
- `check_end`: Checks if the operation has ended.

**Example**:
```python
from swarms.agents import *

# Create an instance of ToolAgent
tool_agent = ToolAgent()

# Check the status of an operation
if check_done():
    print("The operation is done.")
```

**Note**: The specific implementations of the stopping condition functions and the `ToolAgent` class are not detailed in this module, as they are imported from other modules within the `swarms.agents` package.




### `swarms.agents.tool_agent`

**Description**:  
This module defines the `ToolAgent` class, which represents a specialized agent capable of performing tasks using a specified model and tokenizer. It is designed to run operations that require input validation against a JSON schema, generating outputs based on defined tasks.

**Imports**:
- `Any`, `Optional`, `Callable`: Type hints from the `typing` module for flexible parameter types.
- `Agent`: The base class for agents, imported from `swarms.structs.agent`.
- `Jsonformer`: A class responsible for transforming JSON data, imported from `swarms.tools.json_former`.
- `logger`: A logging utility from `swarms.utils.loguru_logger`.

### `ToolAgent`
**Description**:  
Represents a tool agent that performs a specific task using a model and tokenizer. It facilitates the execution of tasks by calling the appropriate model or using the defined JSON schema for structured output.

**Attributes**:  
- `name` (`str`): The name of the tool agent.
- `description` (`str`): A description of what the tool agent does.
- `model` (`Any`): The model used by the tool agent for processing.
- `tokenizer` (`Any`): The tokenizer used by the tool agent to prepare input data.
- `json_schema` (`Any`): The JSON schema that defines the structure of the expected output.
- `max_number_tokens` (`int`): The maximum number of tokens to generate (default is 500).
- `parsing_function` (`Optional[Callable]`): A function for parsing the output, if provided.
- `llm` (`Any`): A language model, if utilized instead of a custom model.

**Methods**:

- `__init__(self, name: str, description: str, model: Any, tokenizer: Any, json_schema: Any, max_number_tokens: int, parsing_function: Optional[Callable], llm: Any, *args, **kwargs) -> None`
    - **Description**: Initializes a new instance of the ToolAgent class.
    - **Parameters**: 
      - `name` (`str`): The name of the tool agent.
      - `description` (`str`): A description of the tool agent.
      - `model` (`Any`): The model to use (if applicable).
      - `tokenizer` (`Any`): The tokenizer to use (if applicable).
      - `json_schema` (`Any`): The JSON schema that outlines the expected output format.
      - `max_number_tokens` (`int`): Maximum token output size.
      - `parsing_function` (`Optional[Callable]`): Optional function to parse the output.
      - `llm` (`Any`): The language model to use as an alternative to a custom model.
      - `*args` and `**kwargs`: Additional arguments and keyword arguments for flexibility.
    - **Return**: None.

- `run(self, task: str, *args, **kwargs) -> Any`
    - **Description**: Executes the tool agent for the specified task, utilizing either a model or a language model based on provided parameters.
    - **Parameters**: 
      - `task` (`str`): The task or prompt to be processed by the tool agent.
      - `*args`: Additional positional arguments for flexibility.
      - `**kwargs`: Additional keyword arguments for flexibility.
    - **Return**: 
      - (`Any`): The output generated by the tool agent based on the input task.
    - **Raises**: 
      - `Exception`: If neither `model` nor `llm` is provided or if an error occurs during task execution.

**Example**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from swarms.agents.tool_agent import ToolAgent

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b")
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")

# Define a JSON schema
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "is_student": {"type": "boolean"},
        "courses": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

# Create and run a ToolAgent
task = "Generate a person's information based on the following schema:"
agent = ToolAgent(model=model, tokenizer=tokenizer, json_schema=json_schema)
generated_data = agent.run(task)

print(generated_data)
```




### `swarms.agents.stopping_conditions`

**Description**:  
This module contains a set of functions that check specific stopping conditions based on strings. These functions return boolean values indicating the presence of certain keywords, which can be used to determine the status of an operation or process.

### Functions:

- `check_done(s: str) -> bool`
    - **Description**: Checks if the string contains the keyword "<DONE>". 
    - **Parameters**: 
      - `s` (`str`): The input string to check.
    - **Return**: 
      - (`bool`): `True` if "<DONE>" is found in the string; otherwise, `False`.

- `check_finished(s: str) -> bool`
    - **Description**: Checks if the string contains the keyword "finished".
    - **Parameters**: 
      - `s` (`str`): The input string to check.
    - **Return**: 
      - (`bool`): `True` if "finished" is found in the string; otherwise, `False`.

- `check_complete(s: str) -> bool`
    - **Description**: Checks if the string contains the keyword "complete".
    - **Parameters**: 
      - `s` (`str`): The input string to check.
    - **Return**: 
      - (`bool`): `True` if "complete" is found in the string; otherwise, `False`.

- `check_success(s: str) -> bool`
    - **Description**: Checks if the string contains the keyword "success".
    - **Parameters**: 
      - `s` (`str`): The input string to check.
    - **Return**: 
      - (`bool`): `True` if "success" is found in the string; otherwise, `False`.

- `check_failure(s: str) -> bool`
    - **Description**: Checks if the string contains the keyword "failure".
    - **Parameters**: 
      - `s` (`str`): The input string to check.
    - **Return**: 
      - (`bool`): `True` if "failure" is found in the string; otherwise, `False`.

- `check_error(s: str) -> bool`
    - **Description**: Checks if the string contains the keyword "error".
    - **Parameters**: 
      - `s` (`str`): The input string to check.
    - **Return**: 
      - (`bool`): `True` if "error" is found in the string; otherwise, `False`.

- `check_stopped(s: str) -> bool`
    - **Description**: Checks if the string contains the keyword "stopped".
    - **Parameters**: 
      - `s` (`str`): The input string to check.
    - **Return**: 
      - (`bool`): `True` if "stopped" is found in the string; otherwise, `False`.

- `check_cancelled(s: str) -> bool`
    - **Description**: Checks if the string contains the keyword "cancelled".
    - **Parameters**: 
      - `s` (`str`): The input string to check.
    - **Return**: 
      - (`bool`): `True` if "cancelled" is found in the string; otherwise, `False`.

- `check_exit(s: str) -> bool`
    - **Description**: Checks if the string contains the keyword "exit".
    - **Parameters**: 
      - `s` (`str`): The input string to check.
    - **Return**: 
      - (`bool`): `True` if "exit" is found in the string; otherwise, `False`.

- `check_end(s: str) -> bool`
    - **Description**: Checks if the string contains the keyword "end".
    - **Parameters**: 
      - `s` (`str`): The input string to check.
    - **Return**: 
      - (`bool`): `True` if "end" is found in the string; otherwise, `False`.

**Example**:
```python
from swarms.agents.stopping_conditions import check_done, check_error

status_message = "The process has finished and <DONE>!"

if check_done(status_message):
    print("The operation is done!")

if check_error(status_message):
    print("An error has occurred!")
``` 

**Note**: Each of these functions provides a simple way to check for specific keywords in a given string, which can be helpful in managing and monitoring tasks or operations.



# Schemas

### `swarms.schemas.base_schemas`

**Description**:  
This module defines various Pydantic models that represent schemas used in machine learning applications. These models facilitate data validation and serialization for different types of content, such as model cards, chat messages, and responses. 

**Imports**:
- `uuid`: A module for generating unique identifiers.
- `time`: A module for time-related functions.
- `List`, `Literal`, `Optional`, `Union`: Type hints from the `typing` module for flexible parameter types.
- `BaseModel`, `Field`: Tools from the `pydantic` module for data validation and settings management.

### `ModelCard`
**Description**:  
A Pydantic model that represents a model card, which provides metadata about a machine learning model.

**Attributes**:  
- `id` (`str`): The unique identifier for the model.
- `object` (`str`): A fixed string indicating the type of object ("model").
- `created` (`int`): The timestamp of model creation, defaults to the current time.
- `owned_by` (`str`): The owner of the model.
- `root` (`Optional[str]`): The root model identifier if applicable.
- `parent` (`Optional[str]`): The parent model identifier if applicable.
- `permission` (`Optional[list]`): A list of permissions associated with the model.

### `ModelList`
**Description**:  
A Pydantic model that represents a list of model cards.

**Attributes**:  
- `object` (`str`): A fixed string indicating the type of object ("list").
- `data` (`List[ModelCard]`): A list containing instances of `ModelCard`.

### `ImageUrl`
**Description**:  
A Pydantic model representing an image URL.

**Attributes**:  
- `url` (`str`): The URL of the image.

### `TextContent`
**Description**:  
A Pydantic model representing text content.

**Attributes**:  
- `type` (`Literal["text"]`): A fixed string indicating the type of content (text).
- `text` (`str`): The actual text content.

### `ImageUrlContent`
**Description**:  
A Pydantic model representing image content via URL.

**Attributes**:  
- `type` (`Literal["image_url"]`): A fixed string indicating the type of content (image URL).
- `image_url` (`ImageUrl`): An instance of `ImageUrl` containing the URL of the image.

### `ContentItem`
**Description**:  
A type alias for a union of `TextContent` and `ImageUrlContent`, representing any content type that can be processed.

### `ChatMessageInput`
**Description**:  
A Pydantic model representing an input message for chat applications.

**Attributes**:  
- `role` (`str`): The role of the sender (e.g., "user", "assistant", or "system").
- `content` (`Union[str, List[ContentItem]]`): The content of the message, which can be a string or a list of content items.

### `ChatMessageResponse`
**Description**:  
A Pydantic model representing a response message in chat applications.

**Attributes**:  
- `role` (`str`): The role of the sender (e.g., "user", "assistant", or "system").
- `content` (`str`, optional): The content of the response message.

### `DeltaMessage`
**Description**:  
A Pydantic model representing a delta update for messages in chat applications.

**Attributes**:  
- `role` (`Optional[Literal["user", "assistant", "system"]]`): The role of the sender, if specified.
- `content` (`Optional[str]`): The content of the delta message, if provided.

### `ChatCompletionRequest`
**Description**:  
A Pydantic model representing a request for chat completion.

**Attributes**:  
- `model` (`str`): The model to use for completing the chat (default is "gpt-4o").
- `messages` (`List[ChatMessageInput]`): A list of input messages for the chat.
- `temperature` (`Optional[float]`): Controls the randomness of the output (default is 0.8).
- `top_p` (`Optional[float]`): An alternative to sampling with temperature (default is 0.8).
- `max_tokens` (`Optional[int]`): The maximum number of tokens to generate (default is 4000).
- `stream` (`Optional[bool]`): If true, the response will be streamed (default is False).
- `repetition_penalty` (`Optional[float]`): A penalty for repeated tokens (default is 1.0).
- `echo` (`Optional[bool]`): If true, the input will be echoed in the output (default is False).

### `ChatCompletionResponseChoice`
**Description**:  
A Pydantic model representing a choice in a chat completion response.

**Attributes**:  
- `index` (`int`): The index of the choice.
- `input` (`str`): The input message.
- `message` (`ChatMessageResponse`): The output message.

### `ChatCompletionResponseStreamChoice`
**Description**:  
A Pydantic model representing a choice in a streamed chat completion response.

**Attributes**:  
- `index` (`int`): The index of the choice.
- `delta` (`DeltaMessage`): The delta update for the message.

### `UsageInfo`
**Description**:  
A Pydantic model representing usage information for a chat completion request.

**Attributes**:  
- `prompt_tokens` (`int`): The number of tokens used in the prompt (default is 0).
- `total_tokens` (`int`): The total number of tokens used (default is 0).
- `completion_tokens` (`Optional[int]`): The number of tokens used in the completion (default is 0).

### `ChatCompletionResponse`
**Description**:  
A Pydantic model representing a response from a chat completion request.

**Attributes**:  
- `model` (`str`): The model used for the completion.
- `object` (`Literal["chat.completion", "chat.completion.chunk"]`): The type of response object.
- `choices` (`List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]`): A list of choices from the completion.
- `created` (`Optional[int]`): The timestamp of when the response was created.

### `AgentChatCompletionResponse`
**Description**:  
A Pydantic model representing a completion response from an agent.

**Attributes**:  
- `id` (`Optional[str]`): The ID of the agent that generated the completion response (default is a new UUID).
- `agent_name` (`Optional[str]`): The name of the agent that generated the response.
- `object` (`Optional[Literal["chat.completion", "chat.completion.chunk"]]`): The type of response object.
- `choices` (`Optional[ChatCompletionResponseChoice]`): The choice from the completion response.
- `created` (`Optional[int]`): The timestamp of when the response was created.

**Example**:
```python
from swarms.schemas.base_schemas import ChatCompletionRequest, ChatMessageInput

# Create a chat completion request
request = ChatCompletionRequest(
    model="gpt-4",
    messages=[
        ChatMessageInput(role="user", content="Hello! How can I help you?")
    ]
)
``` 

**Note**: The Pydantic models in this module provide a structured way to handle data related to machine learning models and chat interactions, ensuring that the data adheres to defined schemas.




### `swarms.schemas.plan`

**Description**:  
This module defines the `Plan` class, which represents a sequence of steps in a structured format. It utilizes Pydantic for data validation and configuration, ensuring that each plan consists of a list of defined steps.

**Imports**:
- `List`: A type hint from the `typing` module for work with lists.
- `BaseModel`: The Pydantic base class for data models, providing validation and serialization features.
- `Step`: A model representing individual steps in the plan, imported from `swarms.schemas.agent_step_schemas`.

### `Plan`
**Description**:  
Represents a sequence of steps that comprise a plan. This class ensures that the data structure adheres to the expected model for steps.

**Attributes**:  
- `steps` (`List[Step]`): A list of steps, where each step is an instance of the `Step` model.

**Config**:
- `orm_mode` (bool): Enables compatibility with ORM models to facilitate data loading from database objects.

**Example**:
```python
from swarms.schemas.plan import Plan
from swarms.schemas.agent_step_schemas import Step

# Create a list of steps
steps = [
    Step(/* initialize step attributes */),
    Step(/* initialize step attributes */),
]

# Create a Plan instance
plan = Plan(steps=steps)

# Access the steps
for step in plan.steps:
    print(step)
```

**Note**: The `Plan` class relies on the `Step` model for its structure, ensuring that the steps in a plan conform to the validation rules defined in the `Step` model.




### `swarms.schemas.__init__`

**Description**:  
This module serves as the initialization point for the schemas subpackage within the Swarms framework. It imports and exposes key classes related to agent steps and agent input schemas, making them available for use in other parts of the application.

**Imports**:
- `Step`: A model representing an individual step in an agent's operation, imported from `swarms.schemas.agent_step_schemas`.
- `ManySteps`: A model representing multiple steps, also imported from `swarms.schemas.agent_step_schemas`.
- `AgentSchema`: A model representing the schema for agent inputs, imported from `swarms.schemas.agent_input_schema`.

**Exported Classes**:
- `Step`: The class for defining individual steps in an agent's operation.
- `ManySteps`: The class for defining multiple steps in an agent's operation.
- `AgentSchema`: The class for defining the input schema for agents.

**Example**:
```python
from swarms.schemas import *

# Create an instance of Step
step = Step(/* initialize step attributes */)

# Create an instance of ManySteps
many_steps = ManySteps(steps=[step, step])

# Create an instance of AgentSchema
agent_schema = AgentSchema(/* initialize agent schema attributes */)
```

**Note**: This module acts as a central point for importing and utilizing the various schema classes defined in the Swarms framework, facilitating structured data handling for agents and their operations.




### `swarms.schemas.agent_step_schemas`

**Description**:  
This module defines the `Step` and `ManySteps` classes, which represent individual steps and collections of steps in a task, respectively. These classes utilize Pydantic for data validation and serialization, ensuring that each step adheres to the defined schema.

**Imports**:
- `time`: A module for time-related functions.
- `uuid`: A module for generating unique identifiers.
- `List`, `Optional`, `Any`: Type hints from the `typing` module for flexible parameter types.
- `BaseModel`, `Field`: Tools from the `pydantic` module for data validation and settings management.
- `AgentChatCompletionResponse`: A model representing the response from an agent's chat completion, imported from `swarms.schemas.base_schemas`.

### `get_current_time() -> str`
**Description**:  
Returns the current time formatted as "YYYY-MM-DD HH:MM:SS".

**Return**:  
- (`str`): The current time as a formatted string.

### `Step`
**Description**:  
A Pydantic model representing a single step in a task, including its ID, completion time, and response from an agent.

**Attributes**:  
- `step_id` (`Optional[str]`): The unique identifier for the step, generated if not provided.
- `time` (`Optional[float]`): The time taken to complete the task step, formatted as a string.
- `response` (`Optional[AgentChatCompletionResponse]`): The response from the agent for this step.

### `ManySteps`
**Description**:  
A Pydantic model representing a collection of steps associated with a specific agent and task.

**Attributes**:  
- `agent_id` (`Optional[str]`): The unique identifier for the agent.
- `agent_name` (`Optional[str]`): The name of the agent.
- `task` (`Optional[str]`): The name of the task being performed.
- `max_loops` (`Optional[Any]`): The maximum number of steps in the task.
- `run_id` (`Optional[str]`): The ID of the task this collection of steps belongs to.
- `steps` (`Optional[List[Step]]`): A list of `Step` instances representing the steps of the task.
- `full_history` (`Optional[str]`): A string containing the full history of the task.
- `total_tokens` (`Optional[int]`): The total number of tokens generated during the task.
- `stopping_token` (`Optional[str]`): The token at which the task stopped.
- `interactive` (`Optional[bool]`): Indicates whether the task is interactive.
- `dynamic_temperature_enabled` (`Optional[bool]`): Indicates whether dynamic temperature adjustments are enabled for the task.

**Example**:
```python
from swarms.schemas.agent_step_schemas import Step, ManySteps

# Create a step instance
step = Step(step_id="12345", response=AgentChatCompletionResponse(...))

# Create a ManySteps instance
many_steps = ManySteps(
    agent_id="agent-1",
    agent_name="Test Agent",
    task="Example Task",
    max_loops=5,
    steps=[step],
    full_history="Task executed successfully.",
    total_tokens=100
)

print(many_steps)
```

**Note**: The `Step` and `ManySteps` classes provide structured representations of task steps, ensuring that all necessary information is captured and validated according to the defined schemas.




### `swarms.schemas.agent_input_schema`

**Description**:  
This module defines the `AgentSchema` class using Pydantic, which represents the input parameters necessary for configuring an agent in the Swarms framework. It includes a variety of attributes for specifying the agent's behavior, model settings, and operational parameters.

**Imports**:
- `Any`, `Callable`, `Dict`, `List`, `Optional`: Type hints from the `typing` module for flexible parameter types.
- `BaseModel`, `Field`: Tools from the `pydantic` module for data validation and settings management.
- `validator`: A decorator from Pydantic used for custom validation of fields.

### `AgentSchema`
**Description**:  
Represents the configuration for an agent, including attributes that govern its behavior, capabilities, and interaction with language models. This class ensures that the input data adheres to defined validation rules.

**Attributes**:  
- `llm` (`Any`): The language model to use.
- `max_tokens` (`int`): The maximum number of tokens the agent can generate, must be greater than or equal to 1.
- `context_window` (`int`): The size of the context window, must be greater than or equal to 1.
- `user_name` (`str`): The name of the user interacting with the agent.
- `agent_name` (`str`): The name of the agent.
- `system_prompt` (`str`): The system prompt provided to the agent.
- `template` (`Optional[str]`): An optional template for the agent, default is `None`.
- `max_loops` (`Optional[int]`): The maximum number of loops the agent can perform (default is 1, must be greater than or equal to 1).
- `stopping_condition` (`Optional[Callable[[str], bool]]`): A callable function that defines a stopping condition for the agent.
- `loop_interval` (`Optional[int]`): The interval between loops (default is 0, must be greater than or equal to 0).
- `retry_attempts` (`Optional[int]`): Number of times to retry an operation if it fails (default is 3, must be greater than or equal to 0).
- `retry_interval` (`Optional[int]`): The time between retry attempts (default is 1, must be greater than or equal to 0).
- `return_history` (`Optional[bool]`): Flag indicating whether to return the history of the agent's operations (default is `False`).
- `stopping_token` (`Optional[str]`): Token indicating when to stop processing (default is `None`).
- `dynamic_loops` (`Optional[bool]`): Indicates whether dynamic loops are enabled (default is `False`).
- `interactive` (`Optional[bool]`): Indicates whether the agent operates in an interactive mode (default is `False`).
- `dashboard` (`Optional[bool]`): Flag indicating whether a dashboard interface is enabled (default is `False`).
- `agent_description` (`Optional[str]`): A description of the agent's functionality (default is `None`).
- `tools` (`Optional[List[Callable]]`): List of callable tools the agent can use (default is `None`).
- `dynamic_temperature_enabled` (`Optional[bool]`): Indicates whether dynamic temperature adjustments are enabled (default is `False`).
- Additional attributes for managing various functionalities and configurations related to the agent's behavior, such as logging, saving states, and managing tools.

### Validators:

- **check_list_items_not_none(v)**: Ensures that items within certain list attributes (`tools`, `docs`, `sop_list`, etc.) are not `None`.
- **check_optional_callable_not_none(v)**: Ensures that optional callable attributes are either `None` or callable.

**Example**:
```python
from swarms.schemas.agent_input_schema import AgentSchema

# Define the agent configuration data
agent_data = {
    "llm": "OpenAIChat",
    "max_tokens": 4096,
    "context_window": 8192,
    "user_name": "Human",
    "agent_name": "test-agent",
    "system_prompt": "Custom system prompt",
}

# Create an AgentSchema instance
agent = AgentSchema(**agent_data)
print(agent)
```

**Note**: The `AgentSchema` class provides a structured way to configure agents in the Swarms framework, ensuring that all necessary parameters are validated before use.


