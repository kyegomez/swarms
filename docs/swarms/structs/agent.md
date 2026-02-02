# `Agent` Structure Reference Documentation

```mermaid
graph TD
    A[Task Initiation] -->|Receives Task| B[Initial LLM Processing]
    B -->|Interprets Task| C[Tool Usage]
    C -->|Calls Tools| D[Function 1]
    C -->|Calls Tools| E[Function 2]
    D -->|Returns Data| C
    E -->|Returns Data| C
    C -->|Provides Data| F[Memory Interaction]
    F -->|Stores and Retrieves Data| G[RAG System]
    G -->|ChromaDB/Pinecone| H[Enhanced Data]
    F -->|Provides Enhanced Data| I[Final LLM Processing]
    I -->|Generates Final Response| J[Output]
    C -->|No Tools Available| K[Skip Tool Usage]
    K -->|Proceeds to Memory Interaction| F
    F -->|No Memory Available| L[Skip Memory Interaction]
    L -->|Proceeds to Final LLM Processing| I
```

The `Agent` class is the core component of the Swarm Agent framework. It serves as an autonomous agent that bridges Language Models (LLMs) with external tools and long-term memory systems. The class is designed to handle a variety of document types—including PDFs, text files, Markdown, and JSON—enabling robust document ingestion and processing. By integrating these capabilities, the `Agent` class empowers LLMs to perform complex tasks, utilize external resources, and manage information efficiently, making it a versatile solution for advanced autonomous workflows.


## Features
The `Agent` class establishes a conversational loop with a language model, allowing for interactive task execution, feedback collection, and dynamic response generation. It includes features such as:

| Feature                                 | Description                                                                                      |
|------------------------------------------|--------------------------------------------------------------------------------------------------|
| **Conversational Loop**                  | Enables back-and-forth interaction with the model.                                               |
| **Feedback Collection**                  | Allows users to provide feedback on generated responses.                                         |
| **Stoppable Conversation**               | Supports custom stopping conditions for the conversation.                                        |
| **Retry Mechanism**                      | Implements a retry system for handling issues in response generation.                            |
| **Tool Integration**                     | Supports the integration of various tools for enhanced capabilities.                             |
| **Long-term Memory Management**          | Incorporates vector databases for efficient information retrieval.                               |
| **Document Ingestion**                   | Processes various document types for information extraction.                                     |
| **Interactive Mode**                     | Allows real-time communication with the agent.                                                   |
| **Sentiment Analysis**                   | Evaluates the sentiment of generated responses.                                                  |
| **Output Filtering and Cleaning**        | Ensures generated responses meet specific criteria.                                              |
| **Asynchronous and Concurrent Execution**| Supports efficient parallelization of tasks.                                                     |
| **Planning and Reasoning**               | Implements planning functionality for enhanced decision-making.                   |
| **Autonomous Planning and Execution**    | When `max_loops="auto"`, automatically creates plans, executes subtasks, and generates summaries. Includes built-in tools for file operations, user communication, and workspace management. |
| **Agent Handoffs and Task Delegation**   | Intelligently routes tasks to specialized agents based on capabilities and task requirements.      |





## `Agent` Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `Optional[str]` | Unique identifier for the agent instance. |
| `llm` | `Optional[Any]` | Language model instance used by the agent. |
| `max_loops` | `Optional[Union[int, str]]` | Maximum number of loops the agent can run. |
| `stopping_condition` | `Optional[Callable[[str], bool]]` | Callable function determining when to stop looping. |
| `loop_interval` | `Optional[int]` | Interval (in seconds) between loops. |
| `retry_attempts` | `Optional[int]` | Number of retry attempts for failed LLM calls. |
| `retry_interval` | `Optional[int]` | Interval (in seconds) between retry attempts. |
| `return_history` | `Optional[bool]` | Boolean indicating whether to return conversation history. |
| `stopping_token` | `Optional[str]` | Token that stops the agent from looping when present in the response. |
| `dynamic_loops` | `Optional[bool]` | Boolean indicating whether to dynamically determine the number of loops. |
| `interactive` | `Optional[bool]` | Boolean indicating whether to run in interactive mode. |
| `dashboard` | `Optional[bool]` | Boolean indicating whether to display a dashboard. |
| `agent_name` | `Optional[str]` | Name of the agent instance. |
| `agent_description` | `Optional[str]` | Description of the agent instance. |
| `system_prompt` | `Optional[str]` | System prompt used to initialize the conversation. |
| `tools` | `List[Callable]` | List of callable functions representing tools the agent can use. |
| `dynamic_temperature_enabled` | `Optional[bool]` | Boolean indicating whether to dynamically adjust the LLM's temperature. |
| `sop` | `Optional[str]` | Standard operating procedure for the agent. |
| `sop_list` | `Optional[List[str]]` | List of strings representing the standard operating procedure. |
| `saved_state_path` | `Optional[str]` | File path for saving and loading the agent's state. |
| `autosave` | `Optional[bool]` | Boolean indicating whether to automatically save the agent's state. |
| `context_length` | `Optional[int]` | Maximum length of the context window (in tokens) for the LLM. |
| `transforms` | `Optional[Union[TransformConfig, dict]]` | Message transformation configuration for handling context limits. |
| `user_name` | `Optional[str]` | Name used to represent the user in the conversation. |
| `multi_modal` | `Optional[bool]` | Boolean indicating whether to support multimodal inputs. |
| `tokenizer` | `Optional[Any]` | Instance of a tokenizer used for token counting and management. |
| `long_term_memory` | `Optional[Union[Callable, Any]]` | Instance of a `BaseVectorDatabase` implementation for long-term memory management. |
| `fallback_model_name` | `Optional[str]` | The fallback model name to use if primary model fails. |
| `fallback_models` | `Optional[List[str]]` | List of model names to try in order. First model is primary, rest are fallbacks. |
| `preset_stopping_token` | `Optional[bool]` | Boolean indicating whether to use a preset stopping token. |
| `streaming_on` | `Optional[bool]` | Boolean indicating whether to stream responses. |
| `stream` | `Optional[bool]` | Boolean indicating whether to enable detailed token-by-token streaming with metadata. |
| `streaming_callback` | `Optional[Callable[[str], None]]` | Callback function to receive streaming tokens in real-time. |
| `verbose` | `Optional[bool]` | Boolean indicating whether to print verbose output. |
| `stopping_func` | `Optional[Callable]` | Callable function used as a stopping condition. |
| `custom_exit_command` | `Optional[str]` | String representing a custom command for exiting the agent's loop. |
| `custom_tools_prompt` | `Optional[Callable]` | Callable function for generating a custom prompt for tool usage. |
| `tool_schema` | `ToolUsageType` | Data structure representing the schema for the agent's tools. |
| `output_type` | `OutputType` | Type representing the expected output type of responses. |
| `function_calling_type` | `str` | String representing the type of function calling. |
| `output_cleaner` | `Optional[Callable]` | Callable function for cleaning the agent's output. |
| `function_calling_format_type` | `Optional[str]` | String representing the format type for function calling. |
| `list_base_models` | `Optional[List[BaseModel]]` | List of base models used for generating tool schemas. |
| `metadata_output_type` | `str` | String representing the output type for metadata. |
| `state_save_file_type` | `str` | String representing the file type for saving the agent's state. |
| `tool_choice` | `str` | String representing the method for tool selection. |
| `rules` | `str` | String representing the rules for the agent's behavior. |
| `planning_prompt` | `Optional[str]` | String representing the prompt for planning. |
| `custom_planning_prompt` | `str` | String representing a custom prompt for planning. |
| `memory_chunk_size` | `int` | Integer representing the maximum size of memory chunks for long-term memory retrieval. |
| `tool_system_prompt` | `str` | String representing the system prompt for tools |
| `max_tokens` | `int` | Integer representing the maximum number of tokens |
| `temperature` | `float` | Float representing the temperature for the LLM |
| `timeout` | `Optional[int]` | Integer representing the timeout for operations in seconds |
| `tags` | `Optional[List[str]]` | Optional list of strings for tagging the agent. |
| `use_cases` | `Optional[List[Dict[str, str]]]` | Optional list of dictionaries describing use cases for the agent. |
| `auto_generate_prompt` | `bool` | Boolean indicating whether to automatically generate prompts. |
| `rag_every_loop` | `bool` | Boolean indicating whether to query RAG database for context on every loop |
| `plan_enabled` | `bool` | Boolean indicating whether planning functionality is enabled |
| `artifacts_on` | `bool` | Boolean indicating whether to save artifacts from agent execution |
| `artifacts_output_path` | `str` | File path where artifacts should be saved |
| `artifacts_file_extension` | `str` | File extension to use for saved artifacts |
| `model_name` | `str` | String representing the name of the model to use |
| `llm_args` | `dict` | Dictionary containing additional arguments for the LLM |
| `load_state_path` | `str` | String representing the path to load state from |
| `role` | `agent_roles` | String representing the role of the agent (e.g., "worker") |
| `print_on` | `bool` | Boolean indicating whether to print output |
| `tools_list_dictionary` | `Optional[List[Dict[str, Any]]]` | List of dictionaries representing tool schemas |
| `mcp_url` | `Optional[Union[str, MCPConnection]]` | String or MCPConnection representing the MCP server URL |
| `mcp_urls` | `List[str]` | List of strings representing multiple MCP server URLs |
| `react_on` | `bool` | Boolean indicating whether to enable ReAct reasoning |
| `safety_prompt_on` | `bool` | Boolean indicating whether to enable safety prompts |
| `random_models_on` | `bool` | Boolean indicating whether to randomly select models |
| `mcp_config` | `Optional[MCPConnection]` | MCPConnection object containing MCP configuration |
| `mcp_configs` | `Optional[MultipleMCPConnections]` | MultipleMCPConnections object for managing multiple MCP server connections |
| `top_p` | `Optional[float]` | Float representing the top-p sampling parameter |
| `llm_base_url` | `Optional[str]` | String representing the base URL for the LLM API |
| `llm_api_key` | `Optional[str]` | String representing the API key for the LLM |
| `tool_call_summary` | `bool` | Boolean indicating whether to summarize tool calls |
| `summarize_multiple_images` | `bool` | Boolean indicating whether to summarize multiple image outputs |
| `tool_retry_attempts` | `int` | Integer representing the number of retry attempts for tool execution |
| `reasoning_prompt_on` | `bool` | Boolean indicating whether to enable reasoning prompts |
| `reasoning_effort` | `Optional[str]` | Reasoning effort level for reasoning-enabled models (e.g., "low", "medium", "high") |
| `reasoning_enabled` | `bool` | Boolean indicating whether to enable reasoning capabilities |
| `thinking_tokens` | `Optional[int]` | Maximum number of thinking tokens for reasoning models |
| `dynamic_context_window` | `bool` | Boolean indicating whether to dynamically adjust context window |
| `show_tool_execution_output` | `bool` | Boolean indicating whether to show tool execution output |
| `workspace_dir` | `str` | String representing the workspace directory for the agent |
| `handoffs` | `Optional[Union[Sequence[Callable], Any]]` | List of Agent instances that can be delegated tasks to. When provided, the agent will use a MultiAgentRouter to intelligently route tasks to the most appropriate specialized agent. |
| `capabilities` | `Optional[List[str]]` | List of strings describing the agent's capabilities. |
| `mode` | `Literal["interactive", "fast", "standard"]` | Execution mode: "interactive" for real-time interaction, "fast" for optimized performance, "standard" for default behavior. |
| `publish_to_marketplace` | `bool` | Boolean indicating whether to publish the agent's prompt to the Swarms marketplace. |
| `marketplace_prompt_id` | `Optional[str]` | Unique UUID identifier of a prompt from the Swarms marketplace. When provided, the agent will automatically fetch and load the prompt as the system prompt. |
| `selected_tools` | `Optional[Union[str, List[str]]]` | Controls which tools are available in autonomous mode (`max_loops="auto"`). Use `"all"` for all tools or provide a list of specific tool names. Available tools: `"create_plan"`, `"think"`, `"subtask_done"`, `"complete_task"`, `"respond_to_user"`, `"create_file"`, `"update_file"`, `"read_file"`, `"list_directory"`, `"delete_file"`, `"run_bash"`, `"create_sub_agent"`, `"assign_task"`. |

## `Agent` Methods

| Method | Description | Inputs | Usage Example |
|--------|-------------|--------|----------------|
| `run(task, img=None, imgs=None, correct_answer=None, streaming_callback=None, *args, **kwargs)` | Runs the autonomous agent loop to complete the given task with enhanced parameters. | `task` (str): The task to be performed.<br>`img` (str, optional): Path to a single image file.<br>`imgs` (List[str], optional): List of image paths for batch processing.<br>`correct_answer` (str, optional): Expected correct answer for validation with automatic retries.<br>`streaming_callback` (Callable, optional): Callback function for real-time token streaming.<br>`*args`, `**kwargs`: Additional arguments. | `response = agent.run("Generate a report on financial performance.")` |
| `run_batched(tasks, imgs=None, *args, **kwargs)` | Runs multiple tasks concurrently in batch mode. | `tasks` (List[str]): List of tasks to run.<br>`imgs` (List[str], optional): List of images to process.<br>`*args`, `**kwargs`: Additional arguments. | `responses = agent.run_batched(["Task 1", "Task 2"])` |
| `run_multiple_images(task, imgs, *args, **kwargs)` | Runs the agent with multiple images using concurrent processing. | `task` (str): The task to perform on each image.<br>`imgs` (List[str]): List of image paths or URLs.<br>`*args`, `**kwargs`: Additional arguments. | `outputs = agent.run_multiple_images("Describe image", ["img1.jpg", "img2.png"])` |
| `continuous_run_with_answer(task, img=None, correct_answer=None, max_attempts=10)` | Runs the agent until the correct answer is provided. | `task` (str): The task to perform.<br>`img` (str, optional): Image to process.<br>`correct_answer` (str): Expected answer.<br>`max_attempts` (int): Maximum attempts. | `response = agent.continuous_run_with_answer("Math problem", correct_answer="42")` |
| `tool_execution_retry(response, loop_count)` | Executes tools with retry logic for handling failures. | `response` (any): Response containing tool calls.<br>`loop_count` (int): Current loop number. | `agent.tool_execution_retry(response, 1)` |
| `__call__(task, img=None, *args, **kwargs)` | Alternative way to call the `run` method. | Same as `run`. | `response = agent("Generate a report on financial performance.")` |
| `parse_and_execute_tools(response, *args, **kwargs)` | Parses the agent's response and executes any tools mentioned in it. | `response` (str): The agent's response to be parsed.<br>`*args`, `**kwargs`: Additional arguments. | `agent.parse_and_execute_tools(response)` |
| `add_memory(message)` | Adds a message to the agent's memory. | `message` (str): The message to add. | `agent.add_memory("Important information")` |
| `plan(task, *args, **kwargs)` | Plans the execution of a task. | `task` (str): The task to plan.<br>`*args`, `**kwargs`: Additional arguments. | `agent.plan("Analyze market trends")` |
| `run_concurrent(task, *args, **kwargs)` | Runs a task concurrently. | `task` (str): The task to run.<br>`*args`, `**kwargs`: Additional arguments. | `response = await agent.run_concurrent("Concurrent task")` |
| `run_concurrent_tasks(tasks, *args, **kwargs)` | Runs multiple tasks concurrently. | `tasks` (List[str]): List of tasks to run.<br>`*args`, `**kwargs`: Additional arguments. | `responses = agent.run_concurrent_tasks(["Task 1", "Task 2"])` |
| `bulk_run(inputs)` | Generates responses for multiple input sets. | `inputs` (List[Dict[str, Any]]): List of input dictionaries. | `responses = agent.bulk_run([{"task": "Task 1"}, {"task": "Task 2"}])` |
| `run_multiple_images(task, imgs, *args, **kwargs)` | Runs the agent with multiple images using concurrent processing. | `task` (str): The task to perform on each image.<br>`imgs` (List[str]): List of image paths or URLs.<br>`*args`, `**kwargs`: Additional arguments. | `outputs = agent.run_multiple_images("Describe image", ["img1.jpg", "img2.png"])` |
| `continuous_run_with_answer(task, img=None, correct_answer=None, max_attempts=10)` | Runs the agent until the correct answer is provided. | `task` (str): The task to perform.<br>`img` (str, optional): Image to process.<br>`correct_answer` (str): Expected answer.<br>`max_attempts` (int): Maximum attempts. | `response = agent.continuous_run_with_answer("Math problem", correct_answer="42")` |
| `save()` | Saves the agent's history to a file. | None | `agent.save()` |
| `load(file_path)` | Loads the agent's history from a file. | `file_path` (str): Path to the file. | `agent.load("agent_history.json")` |
| `graceful_shutdown()` | Gracefully shuts down the system, saving the state. | None | `agent.graceful_shutdown()` |
| `analyze_feedback()` | Analyzes the feedback for issues. | None | `agent.analyze_feedback()` |
| `undo_last()` | Undoes the last response and returns the previous state. | None | `previous_state, message = agent.undo_last()` |
| `add_response_filter(filter_word)` | Adds a response filter to filter out certain words. | `filter_word` (str): Word to filter. | `agent.add_response_filter("sensitive")` |
| `apply_response_filters(response)` | Applies response filters to the given response. | `response` (str): Response to filter. | `filtered_response = agent.apply_response_filters(response)` |
| `filtered_run(task)` | Runs a task with response filtering applied. | `task` (str): Task to run. | `response = agent.filtered_run("Generate a report")` |
| `save_to_yaml(file_path)` | Saves the agent to a YAML file. | `file_path` (str): Path to save the YAML file. | `agent.save_to_yaml("agent_config.yaml")` |
| `get_llm_parameters()` | Returns the parameters of the language model. | None | `llm_params = agent.get_llm_parameters()` |
| `save_state(file_path, *args, **kwargs)` | Saves the current state of the agent to a JSON file. | `file_path` (str): Path to save the JSON file.<br>`*args`, `**kwargs`: Additional arguments. | `agent.save_state("agent_state.json")` |
| `update_system_prompt(system_prompt)` | Updates the system prompt. | `system_prompt` (str): New system prompt. | `agent.update_system_prompt("New system instructions")` |
| `update_max_loops(max_loops)` | Updates the maximum number of loops. | `max_loops` (int): New maximum number of loops. | `agent.update_max_loops(5)` |
| `update_loop_interval(loop_interval)` | Updates the loop interval. | `loop_interval` (int): New loop interval. | `agent.update_loop_interval(2)` |
| `update_retry_attempts(retry_attempts)` | Updates the number of retry attempts. | `retry_attempts` (int): New number of retry attempts. | `agent.update_retry_attempts(3)` |
| `update_retry_interval(retry_interval)` | Updates the retry interval. | `retry_interval` (int): New retry interval. | `agent.update_retry_interval(5)` |
| `reset()` | Resets the agent's memory. | None | `agent.reset()` |
| `ingest_docs(docs, *args, **kwargs)` | Ingests documents into the agent's memory. | `docs` (List[str]): List of document paths.<br>`*args`, `**kwargs`: Additional arguments. | `agent.ingest_docs(["doc1.pdf", "doc2.txt"])` |
| `ingest_pdf(pdf)` | Ingests a PDF document into the agent's memory. | `pdf` (str): Path to the PDF file. | `agent.ingest_pdf("document.pdf")` |
| `receive_message(name, message)` | Receives a message and adds it to the agent's memory. | `name` (str): Name of the sender.<br>`message` (str): Content of the message. | `agent.receive_message("User", "Hello, agent!")` |
| `send_agent_message(agent_name, message, *args, **kwargs)` | Sends a message from the agent to a user. | `agent_name` (str): Name of the agent.<br>`message` (str): Message to send.<br>`*args`, `**kwargs`: Additional arguments. | `response = agent.send_agent_message("AgentX", "Task completed")` |
| `add_tool(tool)` | Adds a tool to the agent's toolset. | `tool` (Callable): Tool to add. | `agent.add_tool(my_custom_tool)` |
| `add_tools(tools)` | Adds multiple tools to the agent's toolset. | `tools` (List[Callable]): List of tools to add. | `agent.add_tools([tool1, tool2])` |
| `remove_tool(tool)` | Removes a tool from the agent's toolset. | `tool` (Callable): Tool to remove. | `agent.remove_tool(my_custom_tool)` |
| `remove_tools(tools)` | Removes multiple tools from the agent's toolset. | `tools` (List[Callable]): List of tools to remove. | `agent.remove_tools([tool1, tool2])` |
| `get_docs_from_doc_folders()` | Retrieves and processes documents from the specified folder. | None | `agent.get_docs_from_doc_folders()` |
| `memory_query(task, *args, **kwargs)` | Queries the long-term memory for relevant information. | `task` (str): The task or query.<br>`*args`, `**kwargs`: Additional arguments. | `result = agent.memory_query("Find information about X")` |
| `sentiment_analysis_handler(response)` | Performs sentiment analysis on the given response. | `response` (str): The response to analyze. | `agent.sentiment_analysis_handler("Great job!")` |
| `count_and_shorten_context_window(history, *args, **kwargs)` | Counts tokens and shortens the context window if necessary. | `history` (str): The conversation history.<br>`*args`, `**kwargs`: Additional arguments. | `shortened_history = agent.count_and_shorten_context_window(history)` |
| `output_cleaner_and_output_type(response, *args, **kwargs)` | Cleans and formats the output based on specified type. | `response` (str): The response to clean and format.<br>`*args`, `**kwargs`: Additional arguments. | `cleaned_response = agent.output_cleaner_and_output_type(response)` |
| `stream_response(response, delay=0.001)` | Streams the response token by token. | `response` (str): The response to stream.<br>`delay` (float): Delay between tokens. | `agent.stream_response("This is a streamed response")` |
| `dynamic_context_window()` | Dynamically adjusts the context window. | None | `agent.dynamic_context_window()` |
| `check_available_tokens()` | Checks and returns the number of available tokens. | None | `available_tokens = agent.check_available_tokens()` |
| `tokens_checks()` | Performs token checks and returns available tokens. | None | `token_info = agent.tokens_checks()` |
| `truncate_string_by_tokens(input_string, limit)` | Truncates a string to fit within a token limit. | `input_string` (str): String to truncate.<br>`limit` (int): Token limit. | `truncated_string = agent.truncate_string_by_tokens("Long string", 100)` |
| `tokens_operations(input_string)` | Performs various token-related operations on the input string. | `input_string` (str): String to process. | `processed_string = agent.tokens_operations("Input string")` |
| `parse_function_call_and_execute(response)` | Parses a function call from the response and executes it. | `response` (str): Response containing the function call. | `result = agent.parse_function_call_and_execute(response)` |
| `llm_output_parser(response)` | Parses the output from the language model. | `response` (Any): Response from the LLM. | `parsed_response = agent.llm_output_parser(llm_output)` |
| `log_step_metadata(loop, task, response)` | Logs metadata for each step of the agent's execution. | `loop` (int): Current loop number.<br>`task` (str): Current task.<br>`response` (str): Agent's response. | `agent.log_step_metadata(1, "Analyze data", "Analysis complete")` |
| `to_dict()` | Converts the agent's attributes to a dictionary. | None | `agent_dict = agent.to_dict()` |
| `to_json(indent=4, *args, **kwargs)` | Converts the agent's attributes to a JSON string. | `indent` (int): Indentation for JSON.<br>`*args`, `**kwargs`: Additional arguments. | `agent_json = agent.to_json()` |
| `to_yaml(indent=4, *args, **kwargs)` | Converts the agent's attributes to a YAML string. | `indent` (int): Indentation for YAML.<br>`*args`, `**kwargs`: Additional arguments. | `agent_yaml = agent.to_yaml()` |
| `to_toml(*args, **kwargs)` | Converts the agent's attributes to a TOML string. | `*args`, `**kwargs`: Additional arguments. | `agent_toml = agent.to_toml()` |
| `model_dump_json()` | Saves the agent model to a JSON file in the workspace directory. | None | `agent.model_dump_json()` |
| `model_dump_yaml()` | Saves the agent model to a YAML file in the workspace directory. | None | `agent.model_dump_yaml()` |
| `log_agent_data()` | Logs the agent's data to an external API. | None | `agent.log_agent_data()` |
| `handle_tool_schema_ops()` | Handles operations related to tool schemas. | None | `agent.handle_tool_schema_ops()` |
| `handle_handoffs(task)` | Handles task delegation to specialized agents when handoffs are configured. | `task` (str): Task to be delegated to appropriate specialized agent. | `response = agent.handle_handoffs("Analyze market data")` |
| `call_llm(task, *args, **kwargs)` | Calls the appropriate method on the language model. | `task` (str): Task for the LLM.<br>`*args`, `**kwargs`: Additional arguments. | `response = agent.call_llm("Generate text")` |
| `handle_sop_ops()` | Handles operations related to standard operating procedures. | None | `agent.handle_sop_ops()` |
| `agent_output_type(responses)` | Processes and returns the agent's output based on the specified output type. | `responses` (list): List of responses. | `formatted_output = agent.agent_output_type(responses)` |
| `check_if_no_prompt_then_autogenerate(task)` | Checks if a system prompt is not set and auto-generates one if needed. | `task` (str): The task to use for generating a prompt. | `agent.check_if_no_prompt_then_autogenerate("Analyze data")` |
| `handle_artifacts(response, output_path, extension)` | Handles saving artifacts from agent execution | `response` (str): Agent response<br>`output_path` (str): Output path<br>`extension` (str): File extension | `agent.handle_artifacts(response, "outputs/", ".txt")` |
| `showcase_config()` | Displays the agent's configuration in a formatted table. | None | `agent.showcase_config()` |
| `talk_to(agent, task, img=None, *args, **kwargs)` | Initiates a conversation with another agent. | `agent` (Any): Target agent.<br>`task` (str): Task to discuss.<br>`img` (str, optional): Image to share.<br>`*args`, `**kwargs`: Additional arguments. | `response = agent.talk_to(other_agent, "Let's collaborate")` |
| `talk_to_multiple_agents(agents, task, *args, **kwargs)` | Talks to multiple agents concurrently. | `agents` (List[Any]): List of target agents.<br>`task` (str): Task to discuss.<br>`*args`, `**kwargs`: Additional arguments. | `responses = agent.talk_to_multiple_agents([agent1, agent2], "Group discussion")` |
| `get_agent_role()` | Returns the role of the agent. | None | `role = agent.get_agent_role()` |
| `pretty_print(response, loop_count)` | Prints the response in a formatted panel. | `response` (str): Response to print.<br>`loop_count` (int): Current loop number. | `agent.pretty_print("Analysis complete", 1)` |
| `parse_llm_output(response)` | Parses and standardizes the output from the LLM. | `response` (Any): Response from the LLM. | `parsed_response = agent.parse_llm_output(llm_output)` |
| `sentiment_and_evaluator(response)` | Performs sentiment analysis and evaluation on the response. | `response` (str): Response to analyze. | `agent.sentiment_and_evaluator("Great response!")` |
| `output_cleaner_op(response)` | Applies output cleaning operations to the response. | `response` (str): Response to clean. | `cleaned_response = agent.output_cleaner_op(response)` |
| `mcp_tool_handling(response, current_loop)` | Handles MCP tool execution and responses. | `response` (Any): Response containing tool calls.<br>`current_loop` (int): Current loop number. | `agent.mcp_tool_handling(response, 1)` |
| `temp_llm_instance_for_tool_summary()` | Creates a temporary LLM instance for tool summaries. | None | `temp_llm = agent.temp_llm_instance_for_tool_summary()` |
| `execute_tools(response, loop_count)` | Executes tools based on the LLM response. | `response` (Any): Response containing tool calls.<br>`loop_count` (int): Current loop number. | `agent.execute_tools(response, 1)` |
| `list_output_types()` | Returns available output types. | None | `types = agent.list_output_types()` |
| `tool_execution_retry(response, loop_count)` | Executes tools with retry logic for handling failures. | `response` (Any): Response containing tool calls.<br>`loop_count` (int): Current loop number. | `agent.tool_execution_retry(response, 1)` |



## `Agent.run(*args, **kwargs)`

The `run` method has been significantly enhanced with new parameters for advanced functionality:

### Method Signature
```python
def run(
    self,
    task: Optional[Union[str, Any]] = None,
    img: Optional[str] = None,
    imgs: Optional[List[str]] = None,
    correct_answer: Optional[str] = None,
    streaming_callback: Optional[Callable[[str], None]] = None,
    *args,
    **kwargs,
) -> Any:
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `task` | `Optional[Union[str, Any]]` | The task to be executed | `None` |
| `img` | `Optional[str]` | Path to a single image file | `None` |
| `imgs` | `Optional[List[str]]` | List of image paths for batch processing | `None` |
| `correct_answer` | `Optional[str]` | Expected correct answer for validation with automatic retries | `None` |
| `streaming_callback` | `Optional[Callable[[str], None]]` | Callback function to receive streaming tokens in real-time | `None` |
| `*args` | `Any` | Additional positional arguments | - |
| `**kwargs` | `Any` | Additional keyword arguments | - |

### Examples


```python
# --- Enhanced Run Method Examples ---

# Basic Usage
# Simple task execution
response = agent.run("Generate a report on financial performance.")

# Single Image Processing
# Process a single image
response = agent.run(
    task="Analyze this image and describe what you see",
    img="path/to/image.jpg"
)

# Multiple Image Processing
# Process multiple images concurrently
response = agent.run(
    task="Analyze these images and identify common patterns",
    imgs=["image1.jpg", "image2.png", "image3.jpeg"]
)

# Answer Validation with Retries
# Run until correct answer is found
response = agent.run(
    task="What is the capital of France?",
    correct_answer="Paris"
)

# Real-time Streaming
def streaming_callback(token: str):
    print(token, end="", flush=True)

response = agent.run(
    task="Tell me a long story about space exploration",
    streaming_callback=streaming_callback
)

# Combined Parameters
# Complex task with multiple features
response = agent.run(
    task="Analyze these financial charts and provide insights",
    imgs=["chart1.png", "chart2.png", "chart3.png"],
    correct_answer="market volatility",
    streaming_callback=my_callback
)
```

### Return Types

The `run` method returns different types based on the input parameters:

| Scenario              | Return Type                                   | Description                                             |
|-----------------------|-----------------------------------------------|---------------------------------------------------------|
| Single task           | `str`                                         | Returns the agent's response                            |
| Multiple images       | `List[Any]`                                   | Returns a list of results, one for each image           |
| Answer validation     | `str`                                         | Returns the correct answer as a string                  |
| Streaming             | `str`                                         | Returns the complete response after streaming completes |



## Advanced Capabilities

### Tool Integration

The `Agent` class allows seamless integration of external tools by accepting a list of Python functions via the `tools` parameter during initialization. Each tool function must include type annotations and a docstring. The `Agent` class automatically converts these functions into an OpenAI-compatible function calling schema, making them accessible for use during task execution.

Learn more about tools [here](https://docs.swarms.world/en/latest/swarms/tools/tools_examples/)

## Requirements for a tool

| Requirement         | Description                                                      |
|---------------------|------------------------------------------------------------------|
| Function            | The tool must be a Python function.                              |
| With types          | The function must have type annotations for its parameters.      |
| With doc strings    | The function must include a docstring describing its behavior.   |
| Must return a string| The function must return a string value.                         |

```python
from swarms import Agent
import subprocess

def terminal(code: str):
    """
    Run code in the terminal.

    Args:
        code (str): The code to run in the terminal.

    Returns:
        str: The output of the code.
    """
    out = subprocess.run(code, shell=True, capture_output=True, text=True).stdout
    return str(out)

# Initialize the agent with a tool
agent = Agent(
    agent_name="Terminal-Agent",
    model_name="claude-sonnet-4-20250514",
    tools=[terminal],
    system_prompt="You are an agent that can execute terminal commands. Use the tools provided to assist the user.",
)

# Run the agent
response = agent.run("List the contents of the current directory")
print(response)
```

### Long-term Memory Management

The Swarm Agent supports integration with vector databases for long-term memory management. Here's an example using ChromaDB:

```python
from swarms import Agent
from swarms_memory import ChromaDB

# Initialize ChromaDB
chromadb = ChromaDB(
    metric="cosine",
    output_dir="finance_agent_rag",
)

# Initialize the agent with long-term memory
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    model_name="claude-sonnet-4-20250514",
    long_term_memory=chromadb,
    system_prompt="You are a financial analysis agent with access to long-term memory.",
)

# Run the agent
response = agent.run("What are the components of a startup's stock incentive equity plan?")
print(response)
```

### Agent Handoffs and Task Delegation

The `Agent` class supports intelligent task delegation through the `handoffs` parameter. When provided with a list of specialized agents, the main agent acts as a router that analyzes incoming tasks and delegates them to the most appropriate specialized agent based on their capabilities and descriptions.

#### How Handoffs Work

1. **Task Analysis**: When a task is received, the main agent uses a built-in "boss agent" to analyze the task requirements
2. **Agent Selection**: The boss agent evaluates all available specialized agents and selects the most suitable one(s) based on their descriptions and capabilities
3. **Task Delegation**: The selected agent(s) receive the task (potentially modified for better execution) and process it
4. **Response Aggregation**: Results from specialized agents are collected and returned

#### Key Features

| Feature                   | Description                                                                                   |
|---------------------------|-----------------------------------------------------------------------------------------------|
| **Intelligent Routing**   | Uses AI to determine the best agent for each task                                             |
| **Multiple Agent Support**| Can delegate to multiple agents for complex tasks requiring different expertise               |
| **Task Modification**     | Can modify tasks to better suit the selected agent's capabilities                             |
| **Transparent Reasoning** | Provides clear explanations for agent selection decisions                                     |
| **Seamless Integration**  | Works transparently with the existing `run()` method                                          |

#### Basic Handoff Example

```python
from swarms.structs.agent import Agent

# Create specialized agents
research_agent = Agent(
    agent_name="ResearchAgent",
    agent_description="Specializes in researching topics and providing detailed, factual information",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a research specialist. Provide detailed, well-researched information about any topic, citing sources when possible.",
)

code_agent = Agent(
    agent_name="CodeExpertAgent",
    agent_description="Expert in writing, reviewing, and explaining code across multiple programming languages",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a coding expert. Write, review, and explain code with a focus on best practices and clean code principles.",
)

writing_agent = Agent(
    agent_name="WritingAgent",
    agent_description="Skilled in creative and technical writing, content creation, and editing",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a writing specialist. Create, edit, and improve written content while maintaining appropriate tone and style.",
)

# Create a coordinator agent with handoffs enabled
coordinator = Agent(
    agent_name="CoordinatorAgent",
    agent_description="Coordinates tasks and delegates to specialized agents",
    model_name="gpt-4o-mini",
    max_loops=1,
    handoffs=[research_agent, code_agent, writing_agent],
    system_prompt="You are a coordinator agent. Analyze tasks and delegate them to the most appropriate specialized agent using the handoff_task tool. You can delegate to multiple agents if needed.",
    output_type="all",
)

# Run task - will be automatically delegated to appropriate agent(s)
task = "Call all the agents available to you and ask them how they are doing"
result = coordinator.run(task=task)
print(result)
```


#### Use Cases

- **Financial Analysis**: Route different types of financial analysis to specialized agents (risk, valuation, market analysis)
- **Software Development**: Delegate coding, testing, documentation, and code review to different agents
- **Research Projects**: Route research tasks to domain-specific agents
- **Customer Support**: Delegate different types of inquiries to specialized support agents
- **Content Creation**: Route writing, editing, and fact-checking to different content specialists

### Interactive Mode

To enable interactive mode, set the `interactive` parameter to `True` when initializing the `Agent`. See the Examples section for a complete code example.


### Batch Processing with `run_batched`

The new `run_batched` method allows you to process multiple tasks efficiently:

#### Method Signature

```python
def run_batched(
    self,
    tasks: List[str],
    imgs: List[str] = None,
    *args,
    **kwargs,
) -> List[Any]:
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tasks` | `List[str]` | List of tasks to run concurrently | Required |
| `imgs` | `List[str]` | List of images to process with each task | `None` |
| `*args` | `Any` | Additional positional arguments | - |
| `**kwargs` | `Any` | Additional keyword arguments | - |

#### Usage Examples

```python
# Process multiple tasks in batch
tasks = [
    "Analyze the financial data for Q1",
    "Generate a summary report for stakeholders", 
    "Create recommendations for Q2 planning"
]

# Run all tasks concurrently
batch_results = agent.run_batched(tasks)

# Process results
for i, (task, result) in enumerate(zip(tasks, batch_results)):
    print(f"Task {i+1}: {task}")
    print(f"Result: {result}\n")
```

#### Batch Processing with Images

```python
# Process multiple tasks with multiple images
tasks = [
    "Analyze this chart for trends",
    "Identify patterns in this data visualization",
    "Summarize the key insights from this graph"
]

images = ["chart1.png", "chart2.png", "chart3.png"]

# Each task will process all images
batch_results = agent.run_batched(tasks, imgs=images)
```

#### Return Type

- **Returns**: `List[Any]` - List of results from each task execution
- **Order**: Results are returned in the same order as the input tasks

### Various other settings

```python
# # Convert the agent object to a dictionary
print(agent.to_dict())
print(agent.to_toml())
print(agent.model_dump_json())
print(agent.model_dump_yaml())

# Ingest documents into the agent's knowledge base
agent.ingest_docs("your_pdf_path.pdf")

# Receive a message from a user and process it
agent.receive_message(name="agent_name", message="message")

# Send a message from the agent to a user
agent.send_agent_message(agent_name="agent_name", message="message")

# Ingest multiple documents into the agent's knowledge base
agent.ingest_docs("your_pdf_path.pdf", "your_csv_path.csv")

# Run the agent with a filtered system prompt
agent.filtered_run(
    "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?"
)

# Run the agent with multiple system prompts
agent.bulk_run(
    [
        "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?",
        "Another system prompt",
    ]
)

# Add a memory to the agent
agent.add_memory("Add a memory to the agent")

# Check the number of available tokens for the agent
agent.check_available_tokens()

# Perform token checks for the agent
agent.tokens_checks()

# Print the dashboard of the agent
agent.print_dashboard()


# Fetch all the documents from the doc folders
agent.get_docs_from_doc_folders()

# Dump the model to a JSON file
agent.model_dump_json()
print(agent.to_toml())
```


## Examples

### Tool Integration Examples

#### Basic Tool Example

Create a custom tool function and integrate it with an agent. The agent can then use the tool to execute terminal commands, extending its capabilities beyond text generation.

```python
from swarms import Agent
import subprocess

def terminal(code: str):
    """
    Run code in the terminal.

    Args:
        code (str): The code to run in the terminal.

    Returns:
        str: The output of the code.
    """
    out = subprocess.run(code, shell=True, capture_output=True, text=True).stdout
    return str(out)

# Initialize the agent with a tool
agent = Agent(
    agent_name="Terminal-Agent",
    model_name="claude-sonnet-4-20250514",
    tools=[terminal],
    system_prompt="You are an agent that can execute terminal commands. Use the tools provided to assist the user.",
)

# Run the agent
response = agent.run("List the contents of the current directory")
print(response)
```

#### Agent Structured Outputs with Tools

Use structured tool schemas (OpenAI function calling format) with an agent. The agent receives tool definitions as dictionaries and can call them to retrieve structured data, such as stock prices. The output can be converted from string to dictionary format for easier processing.

```python
from dotenv import load_dotenv
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms.utils.str_to_dict import str_to_dict

load_dotenv()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Retrieve the current stock price and related information for a specified company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol of the company, e.g. AAPL for Apple Inc.",
                    },
                    "include_history": {
                        "type": "boolean",
                        "description": "Indicates whether to include historical price data along with the current price.",
                    },
                    "time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Optional parameter to specify the time for which the stock data is requested, in ISO 8601 format.",
                    },
                },
                "required": [
                    "ticker",
                    "include_history",
                    "time",
                ],
            },
        },
    }
]

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    tools_list_dictionary=tools,
)

out = agent.run(
    "What is the current stock price for Apple Inc. (AAPL)? Include historical price data.",
)

print(out)
print(type(out))
print(str_to_dict(out))
print(type(str_to_dict(out)))
```

#### Long-term Memory with Tools

Integrate a vector database (ChromaDB) with an agent for long-term memory management. The agent can store and retrieve information from the vector database, enabling it to access previously learned knowledge and provide more contextually relevant responses.

```python
from swarms import Agent
from swarms_memory import ChromaDB

# Initialize ChromaDB
chromadb = ChromaDB(
    metric="cosine",
    output_dir="finance_agent_rag",
)

# Initialize the agent with long-term memory
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    model_name="claude-sonnet-4-20250514",
    long_term_memory=chromadb,
    system_prompt="You are a financial analysis agent with access to long-term memory.",
)

# Run the agent
response = agent.run("What are the components of a startup's stock incentive equity plan?")
print(response)
```

### Handoffs Examples

#### Basic Handoff Example

Set up a multi-agent system with task delegation. Three specialized agents (ResearchAgent, CodeExpertAgent, and WritingAgent) are created, and a coordinator agent intelligently routes tasks to the most appropriate specialized agent(s) based on the task requirements. The coordinator can delegate to multiple agents if needed.

```python
from swarms.structs.agent import Agent

# Create specialized agents
research_agent = Agent(
    agent_name="ResearchAgent",
    agent_description="Specializes in researching topics and providing detailed, factual information",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a research specialist. Provide detailed, well-researched information about any topic, citing sources when possible.",
)

code_agent = Agent(
    agent_name="CodeExpertAgent",
    agent_description="Expert in writing, reviewing, and explaining code across multiple programming languages",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a coding expert. Write, review, and explain code with a focus on best practices and clean code principles.",
)

writing_agent = Agent(
    agent_name="WritingAgent",
    agent_description="Skilled in creative and technical writing, content creation, and editing",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a writing specialist. Create, edit, and improve written content while maintaining appropriate tone and style.",
)

# Create a coordinator agent with handoffs enabled
coordinator = Agent(
    agent_name="CoordinatorAgent",
    agent_description="Coordinates tasks and delegates to specialized agents",
    model_name="gpt-4o-mini",
    max_loops=1,
    handoffs=[research_agent, code_agent, writing_agent],
    system_prompt="You are a coordinator agent. Analyze tasks and delegate them to the most appropriate specialized agent using the handoff_task tool. You can delegate to multiple agents if needed.",
    output_type="all",
)

# Run task - will be automatically delegated to appropriate agent(s)
task = "Call all the agents available to you and ask them how they are doing"
result = coordinator.run(task=task)
print(result)
```

### Autonomous Agent Examples

#### Autonomous Agent with Automatic Planning

The autonomous agent mode uses `max_loops="auto"` to enable automatic planning and execution. The agent creates a structured plan with subtasks, executes them sequentially with dependency management, and generates a comprehensive summary. Ideal for complex tasks that require multi-step reasoning and planning, such as generating comprehensive financial reports.

The Agent supports autonomous operation with automatic planning when `max_loops="auto"` is set. This enables the agent to create a plan, execute subtasks, and generate a comprehensive summary automatically.

```python
from swarms.structs.agent import Agent

# Initialize the agent with autonomous mode
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    max_loops="auto",  # Enable autonomous planning and execution
    dynamic_context_window=True,
    top_p=None,
    output_type="all",
)

# Define a complex task that requires planning
quant_report_prompt = (
    "You are an expert in quantitative trading and financial analysis. "
    "Please generate a comprehensive, data-driven report on the top 5 publicly traded energy stocks as of today. "
    "For each stock, include the following: \n"
    "- Company name and ticker\n"
    "- Brief business overview\n"
    "- Key financial metrics (such as market cap, P/E ratio, recent performance)\n"
    "- Recent news or notable events influencing the stock\n"
    "- A concise analysis of why it is currently considered a top energy stock\n"
    "Present your findings in a clear, organized format suitable for professional review."
    "Only create 3 subtasks in your plan, make it very simple"
)

# Run the agent - it will automatically:
# 1. Create a plan with subtasks
# 2. Execute each subtask
# 3. Generate a comprehensive summary
out = agent.run(quant_report_prompt)
print(out)
```

**Key Features of Autonomous Mode:**
- **Automatic Planning**: The agent creates a structured plan with subtasks
- **Subtask Execution**: Each subtask is executed sequentially with dependency management
- **Comprehensive Summary**: Final summary includes all subtask results and insights
- **Error Handling**: Built-in retry logic and error recovery for robust execution
- **Built-in Tools**: Access to file operations, user communication, and workspace management tools

#### Available Tools in Autonomous Mode

When `max_loops="auto"` and `interactive=False`, the agent has access to specialized tools for task execution:

| Tool | Description | Parameters |
|------|-------------|------------|
| `respond_to_user` | Send messages to the user with types (info, question, warning, error, success) | `message` (str), `message_type` (str, optional) |
| `create_file` | Create a new file with specified content | `file_path` (str), `content` (str) |
| `update_file` | Update an existing file with new content | `file_path` (str), `content` (str), `mode` (str: "replace" or "append") |
| `read_file` | Read the contents of a file | `file_path` (str) |
| `list_directory` | List files and directories in a specified path | `directory_path` (str, optional) |
| `delete_file` | Delete a file (with safety checks) | `file_path` (str) |
| `run_bash` | Execute a bash/shell command on the terminal (returns stdout/stderr) | `command` (str), `timeout_seconds` (int, optional, default 60) |
| `create_sub_agent` | Create specialized sub-agents for task delegation | `agents` (array): list of agent specs with `agent_name` (str), `agent_description` (str), `system_prompt` (str, optional) |
| `assign_task` | Assign tasks to sub-agents for asynchronous execution | `assignments` (array): list with `agent_id` (str), `task` (str), `task_id` (str, optional); `wait_for_completion` (bool, optional) |

**File Operations and Workspace Directory:**

All file operations use the agent's workspace directory structure:
- **Workspace Location**: Set via `WORKSPACE_DIR` environment variable (defaults to `agent_workspace` if not set)
- **Agent-Specific Directory**: Each agent gets its own workspace at `workspace_dir/agents/{agent-name}-{uuid}/`
- **Relative Paths**: File paths provided to tools are relative to the agent's workspace directory
- **Absolute Paths**: You can also use absolute paths for files outside the workspace

**Example: Using Autonomous Tools**

```python
from swarms.structs.agent import Agent

# Initialize autonomous agent
agent = Agent(
    agent_name="File-Management-Agent",
    model_name="gpt-4.1",
    max_loops="auto",
    interactive=False,
)

# The agent can now use built-in tools during execution
task = """
Create a comprehensive report on renewable energy trends.
1. Research current trends
2. Create a markdown file with your findings
3. Update the file with additional insights
4. Read the file to verify content
5. Send a message to the user when complete
"""

response = agent.run(task)
# The agent will automatically:
# - Create files in workspace_dir/agents/file-management-agent-{uuid}/
# - Use create_file, update_file, read_file tools as needed
# - Communicate with respond_to_user tool
```

#### Sub-Agent Delegation in Autonomous Mode

The autonomous agent can create and manage sub-agents for task delegation and parallel execution. This enables the main agent to break down complex tasks and distribute work across specialized sub-agents.

**How Sub-Agent Delegation Works:**

1. **Creating Sub-Agents**: The main agent uses `create_sub_agent` tool to create specialized agents with specific roles
2. **Task Assignment**: Tasks are assigned to sub-agents using `assign_task` tool
3. **Parallel Execution**: Sub-agents execute tasks concurrently using asyncio
4. **Result Aggregation**: The main agent collects and synthesizes results from all sub-agents

**Sub-Agent Tools:**

| Tool | Purpose | Key Features |
|------|---------|--------------|
| `create_sub_agent` | Create specialized sub-agents | - Each sub-agent gets unique ID<br>- Cached for reuse<br>- Inherits parent's LLM config<br>- Optional custom system prompt |
| `assign_task` | Delegate tasks to sub-agents | - Asynchronous execution<br>- Multiple task assignments<br>- Wait or fire-and-forget modes<br>- Detailed result reporting |

**Example: Autonomous Agent with Sub-Agent Delegation**

```python
from swarms.structs.agent import Agent

# Initialize autonomous agent that will use sub-agents
coordinator = Agent(
    agent_name="Research-Coordinator",
    agent_description="Coordinates complex research by delegating to specialized sub-agents",
    model_name="gpt-4.1",
    max_loops="auto",
    selected_tools="all",  # Enable all autonomous tools including sub-agent tools
)

# Complex task requiring parallel research
task = """
Conduct comprehensive research on three emerging technology trends:
1. Artificial Intelligence in Healthcare
2. Quantum Computing Advances
3. Renewable Energy Innovations

For each topic:
- Create a specialized sub-agent expert in that domain
- Assign research tasks to each sub-agent to work in parallel
- Compile all findings into a comprehensive report

Use the create_sub_agent and assign_task tools to distribute the work efficiently.
"""

# Run the coordinator - it will automatically:
# 1. Create 3 specialized sub-agents (one for each research area)
# 2. Assign tasks to them using assign_task
# 3. Wait for all sub-agents to complete their work
# 4. Aggregate and synthesize the results
result = coordinator.run(task)
print(result)
```

**What Happens During Execution:**

```python
# The agent internally calls these tools:

# Step 1: Create specialized sub-agents
create_sub_agent({
    "agents": [
        {
            "agent_name": "AI-Healthcare-Expert",
            "agent_description": "Expert in AI applications in healthcare and medical technology",
            "system_prompt": "You are a research expert specializing in AI healthcare applications..."
        },
        {
            "agent_name": "Quantum-Computing-Expert",
            "agent_description": "Expert in quantum computing research and development",
            "system_prompt": "You are a quantum computing research specialist..."
        },
        {
            "agent_name": "Renewable-Energy-Expert",
            "agent_description": "Expert in renewable energy technologies and innovations"
        }
    ]
})
# Returns: "Successfully created 3 sub-agent(s): AI-Healthcare-Expert (ID: sub-agent-a1b2c3d4), ..."

# Step 2: Assign tasks to sub-agents
assign_task({
    "assignments": [
        {
            "agent_id": "sub-agent-a1b2c3d4",
            "task": "Research the latest AI applications in healthcare, focusing on diagnostics and patient care",
            "task_id": "healthcare-research"
        },
        {
            "agent_id": "sub-agent-e5f6g7h8",
            "task": "Analyze recent breakthroughs in quantum computing and their practical applications",
            "task_id": "quantum-research"
        },
        {
            "agent_id": "sub-agent-i9j0k1l2",
            "task": "Investigate the latest innovations in renewable energy technologies",
            "task_id": "energy-research"
        }
    ],
    "wait_for_completion": true
})
# Returns: Detailed results from all three sub-agents
```

**Sub-Agent Tool Parameters:**

**`create_sub_agent` Parameters:**
- `agents` (array, required): List of sub-agent specifications
  - `agent_name` (string, required): Name of the sub-agent
  - `agent_description` (string, required): Description of the sub-agent's role and capabilities
  - `system_prompt` (string, optional): Custom system prompt for the sub-agent

**`assign_task` Parameters:**
- `assignments` (array, required): List of task assignments
  - `agent_id` (string, required): ID of the sub-agent (returned from `create_sub_agent`)
  - `task` (string, required): Task description for the sub-agent
  - `task_id` (string, optional): Unique identifier for tracking this task
- `wait_for_completion` (boolean, optional): Whether to wait for all tasks to complete (default: true)

**Benefits of Sub-Agent Delegation:**

| Benefit | Description |
|---------|-------------|
| **Parallel Processing** | Multiple tasks execute simultaneously for faster completion |
| **Specialization** | Each sub-agent can focus on a specific domain or capability |
| **Scalability** | Complex tasks can be broken into manageable pieces |
| **Reusability** | Sub-agents are cached and can handle multiple assignments |
| **Fault Tolerance** | One sub-agent failure doesn't stop others from completing |

### Loop Examples

#### Multiple Loops Example

Configure an agent with multiple reasoning loops. By setting `max_loops=3` and enabling `reasoning_prompt_on`, the agent performs iterative reasoning, allowing it to refine its thinking over multiple iterations. Useful for complex problems that require step-by-step analysis.

```python
from swarms import Agent

# Agent with multiple loops for iterative reasoning
agent = Agent(
    agent_name="Iterative-Reasoning-Agent",
    model_name="gpt-4.1",
    max_loops=3,  # Run 3 reasoning loops
    reasoning_prompt_on=True,
    system_prompt="You are an agent that reasons through problems step by step.",
)

response = agent.run("Solve this complex problem step by step: [problem description]")
print(response)
```

#### Dynamic Loops Example

Dynamic loop configuration allows the agent to automatically determine the optimal number of reasoning loops based on task complexity. By setting `dynamic_loops=True`, the agent adapts its reasoning depth, using more loops for complex tasks and fewer for simple ones.

```python
from swarms import Agent

# Agent with dynamic loops that adjust based on task complexity
agent = Agent(
    agent_name="Dynamic-Agent",
    model_name="gpt-4.1",
    dynamic_loops=True,  # Automatically determines number of loops
    system_prompt="You are an adaptive agent that adjusts your reasoning depth based on task complexity.",
)

response = agent.run("Analyze this complex scenario and provide insights")
print(response)
```

### Simple Examples

#### Basic Agent Usage

The simplest way to use an agent with minimal configuration. Only requires a model name and max_loops parameter. Perfect for getting started quickly with basic text generation tasks.

```python
from swarms import Agent

# Simple agent with minimal configuration
agent = Agent(
    model_name="gpt-4o-mini",
    max_loops=1,
)

response = agent.run("What is the capital of France?")
print(response)
```

#### Interactive Mode

Enable interactive mode for real-time conversation with the agent. When `interactive=True`, the agent prompts for user input after each response, creating a conversational loop. Useful for interactive applications, chatbots, or when you need to guide the agent through a multi-turn conversation.

```python
from swarms import Agent

# Agent with interactive mode enabled
agent = Agent(
    agent_name="Interactive-Agent",
    model_name="claude-sonnet-4-20250514",
    interactive=True,
    system_prompt="You are an interactive agent. Engage in a conversation with the user.",
)

# Run the agent in interactive mode
agent.run("Let's start a conversation")
```

#### Auto Generate Prompt Example

Automatic prompt generation creates optimized system prompts without manual engineering. When `auto_generate_prompt=True` and no system prompt is provided, the agent automatically generates a contextually appropriate prompt based on the agent name, description, and task. This feature uses AI to create prompts, reducing the need for manual prompt engineering.

```python
import os
from swarms import Agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the agent with automated prompt engineering enabled
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt=None,  # System prompt is dynamically generated
    model_name="gpt-4.1",
    agent_description=None,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=False,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent.json",
    user_name="Human:",
    output_type="string",
    streaming_on=False,
    auto_generate_prompt=True,  # Enable automated prompt engineering
)

# Run the agent with a task description
agent.run(
    "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria",
)

# Print the dynamically generated system prompt
print(agent.system_prompt)
```

#### Token-by-Token Streaming

Enable detailed token-by-token streaming with metadata. When `stream=True`, the agent streams each token as it's generated, providing real-time feedback and detailed metadata including token count, model information, citations, and usage statistics. Useful for building interactive UIs or monitoring agent behavior in real-time.

```python
from swarms import Agent

# Initialize agent with detailed streaming
agent = Agent(
    model_name="gpt-4.1",
    max_loops=1,
    stream=True,  # Enable detailed token-by-token streaming
)

# Run with detailed streaming - each token shows metadata
agent.run("Tell me a short story about a robot learning to paint.")
```

#### Undo Functionality

Revert the agent's last response and restore the previous conversation state. Useful for correcting mistakes, exploring alternative responses, or implementing undo/redo functionality in applications.

```python
# Undo functionality
response = agent.run("Another task")
print(f"Response: {response}")
previous_state, message = agent.undo_last()
print(message)
```

#### Response Filtering

Filter specific words or phrases from agent responses. By adding response filters, you can automatically replace or remove sensitive content, profanity, or unwanted terms from the agent's output. Useful for content moderation, compliance, or customizing output formatting.

```python
# Response filtering
agent.add_response_filter("report")
response = agent.filtered_run("Generate a report on finance")
print(response)
```

#### Saving and Loading State

Save and restore agent state to persist conversations and configurations across sessions. The agent can save its current state to a JSON file and load it later to continue from where it left off. Essential for long-running tasks, debugging, or maintaining conversation continuity.

```python
# Save the agent state
agent.save_state('saved_flow.json')

# Load the agent state
agent = Agent(model_name="gpt-4o-mini", max_loops=5)
agent.load('saved_flow.json')
agent.run("Continue with the task")
```

#### Autosave Functionality

The agent supports automatic saving of configuration and state when `autosave=True`. This ensures your work is preserved even if the agent encounters errors or is interrupted.

**How Autosave Works:**

- **Configuration Saving**: At each loop step, the agent saves its configuration to `workspace_dir/agents/{agent-name}-{uuid}/config.json`
- **State Saving**: Full agent state is saved on errors, interruptions, or when explicitly called
- **Workspace Structure**: Each agent gets its own isolated workspace directory
- **Atomic Writes**: Files are written atomically (via temporary files) to prevent corruption
- **Metadata Tracking**: Each save includes metadata (timestamp, loop count, agent ID)

**Autosave Configuration:**

```python
from swarms import Agent

# Enable autosave
agent = Agent(
    model_name="gpt-4o-mini",
    agent_name="autosave-demo",
    max_loops=5,
    autosave=True,  # Enable automatic saving
    verbose=True,
)

# Run task - configuration is saved at each step
response = agent.run("Complete a complex multi-step task")

# Access the agent's workspace directory
workspace = agent._get_agent_workspace_dir()
print(f"Files saved to: {workspace}")

# Files created:
# - config.json: Agent configuration at each step
# - {agent_name}_state.json: Full agent state
```

**Autosave File Structure:**

```
workspace_dir/
└── agents/
    └── {agent-name}-{uuid}/
        ├── config.json          # Configuration saved at each step
        ├── {agent_name}_state.json  # Full state on save()
        └── [other files created by agent tools]
```

**Workspace Directory Configuration:**

The workspace directory is controlled by the `WORKSPACE_DIR` environment variable:

```python
import os

# Set workspace directory via environment variable
os.environ["WORKSPACE_DIR"] = "/path/to/my/workspace"

# Or it defaults to 'agent_workspace' in current directory
agent = Agent(
    model_name="gpt-4o-mini",
    autosave=True,
)

# Each agent gets its own subdirectory
# workspace_dir/agents/{agent-name}-{uuid}/
```

**Note:** The `workspace_dir` parameter in Agent initialization is ignored. The workspace is always read from the `WORKSPACE_DIR` environment variable, ensuring consistent file organization across all agents.

#### Async and Concurrent Execution

Run tasks asynchronously or in parallel for improved performance. The agent supports concurrent execution of multiple tasks, batch processing, and async operations. Ideal for processing large datasets, handling multiple requests simultaneously, or optimizing throughput in production environments.

```python
# Run a task concurrently
response = await agent.run_concurrent("Concurrent task")
print(response)

# Run multiple tasks concurrently
tasks = [
    {"task": "Task 1"},
    {"task": "Task 2", "img": "path/to/image.jpg"},
    {"task": "Task 3", "custom_param": 42}
]
responses = agent.bulk_run(tasks)
print(responses)

# Run multiple tasks in batch mode
task_list = ["Analyze data", "Generate report", "Create summary"]
batch_responses = agent.run_batched(task_list)
print(f"Completed {len(batch_responses)} tasks in batch mode")
```

## Comprehensive Agent Configuration Examples

### Advanced Agent with All New Features

A comprehensive example showcasing an agent configured with multiple advanced features including memory management, reasoning, MCP integration, artifacts, and batch processing. This demonstrates how to combine various capabilities for production-ready applications.

```python
from swarms import Agent
from swarms_memory import ChromaDB

# Initialize advanced agent with comprehensive configuration
agent = Agent(
    # Basic Configuration
    agent_name="Advanced-Analysis-Agent",
    agent_description="Multi-modal analysis agent with advanced capabilities",
    system_prompt="You are an advanced analysis agent capable of processing multiple data types.",
    
    # Enhanced Run Parameters
    max_loops=3,
    dynamic_loops=True,
    interactive=False,
    dashboard=True,
    
    # Memory and Context Management
    context_length=100000,
    memory_chunk_size=3000,
    dynamic_context_window=True,
    rag_every_loop=True,
    
    # Advanced Features
    auto_generate_prompt=True,
    plan_enabled=True,
    react_on=True,
    safety_prompt_on=True,
    reasoning_prompt_on=True,
    
    # Tool Management
    tool_retry_attempts=5,
    tool_call_summary=True,
    show_tool_execution_output=True,
    function_calling_format_type="OpenAI",
    
    # Artifacts and Output
    artifacts_on=True,
    artifacts_output_path="./outputs",
    artifacts_file_extension=".md",
    output_type="json",
    
    # LLM Configuration
    model_name="gpt-4.1",
    temperature=0.3,
    max_tokens=8000,
    top_p=0.95,
    
    # MCP Integration
    mcp_url="http://localhost:8000",
    mcp_config=None,
    
    # Performance Settings
    timeout=300,
    retry_attempts=3,
    retry_interval=2,
    
    # Metadata and Organization
    tags=["analysis", "multi-modal", "advanced"],
    use_cases=[{"name": "Data Analysis", "description": "Process and analyze complex datasets"}],
    
    # Verbosity and Logging
    verbose=True,
    print_on=True
)

# Run with multiple images and streaming
def streaming_callback(token: str):
    print(token, end="", flush=True)

response = agent.run(
    task="Analyze these financial charts and provide comprehensive insights",
    imgs=["chart1.png", "chart2.png", "chart3.png"],
    streaming_callback=streaming_callback
)

# Run batch processing
tasks = [
    "Analyze Q1 financial performance",
    "Generate Q2 projections",
    "Create executive summary"
]

batch_results = agent.run_batched(tasks)

# Run with answer validation
validated_response = agent.run(
    task="What is the current market trend?",
    correct_answer="bullish",
    max_attempts=5
)
```

### MCP-Enabled Agent Example

Connect an agent to Model Context Protocol (MCP) servers to access external tools and resources. The agent can connect to single or multiple MCP servers, enabling integration with external APIs, databases, and services through a standardized protocol.

```python
from swarms import Agent
from swarms.schemas.mcp_schemas import MCPConnection

# Configure MCP connection
mcp_config = MCPConnection(
    server_path="http://localhost:8000",
    server_name="my_mcp_server",
    capabilities=["tools", "filesystem"]
)

# Initialize agent with MCP integration
mcp_agent = Agent(
    agent_name="MCP-Enabled-Agent",
    system_prompt="You are an agent with access to external tools via MCP.",
    mcp_config=mcp_config,
    mcp_urls=["http://localhost:8000", "http://localhost:8001"],
    tool_call_summary=True
)

# Run with MCP tools
response = mcp_agent.run("Use the available tools to analyze the current system status")
```

### Multi-Image Processing Agent

Process multiple images concurrently and automatically summarize the results. The agent can analyze multiple images in parallel and generate a comprehensive summary of findings across all images, making it ideal for batch image analysis tasks.

```python
# Initialize agent optimized for image processing
image_agent = Agent(
    agent_name="Image-Analysis-Agent",
    system_prompt="You are an expert at analyzing images and extracting insights.",
    multi_modal=True,
    summarize_multiple_images=True,
    artifacts_on=True,
    artifacts_output_path="./image_analysis",
    artifacts_file_extension=".txt"
)

# Process multiple images with summarization
images = ["product1.jpg", "product2.jpg", "product3.jpg"]
analysis = image_agent.run(
    task="Analyze these product images and identify design patterns",
    imgs=images
)

# The agent will automatically summarize results if summarize_multiple_images=True
print(f"Analysis complete: {len(analysis)} images processed")
```

## Simple Examples for New Features

### Fallback Models

Configure multiple models as fallbacks for improved reliability. If the primary model fails, the agent automatically switches to the next model in the fallback list. Ensures task completion even when individual models encounter errors or rate limits.

```python
from swarms import Agent

# Agent with fallback models - automatically switches if primary fails
agent = Agent(
    model_name="gpt-4o",
    fallback_models=["gpt-4o-mini", "gpt-3.5-turbo"],
    max_loops=1
)

# Will try gpt-4o first, then fallback to gpt-4o-mini if it fails
response = agent.run("Analyze this data")
```

### Marketplace Prompt Loading

Load pre-built prompts directly from the Swarms marketplace using a prompt ID. The agent automatically fetches and applies the prompt as its system prompt, enabling one-line prompt loading without manual configuration. Requires the SWARMS_API_KEY environment variable.

```python
from swarms import Agent

# Load a prompt from the Swarms marketplace
agent = Agent(
    model_name="gpt-4o-mini",
    marketplace_prompt_id="550e8400-e29b-41d4-a716-446655440000",
    max_loops=1
)

# Agent automatically loads the system prompt from marketplace
response = agent.run("Execute the marketplace prompt task")
```

### Reasoning-Enabled Models

Configure agents to use reasoning-enabled models like o1-preview. These models perform internal reasoning before generating responses, making them ideal for complex mathematical problems, logical puzzles, and tasks requiring deep analytical thinking. Control reasoning effort and thinking token limits.

```python
from swarms import Agent

# Agent with reasoning capabilities
agent = Agent(
    model_name="o1-preview",
    reasoning_enabled=True,
    reasoning_effort="high",
    thinking_tokens=10000,
    max_loops=1
)

response = agent.run("Solve this complex mathematical problem step by step")
```

### Execution Modes

Choose from three execution modes to optimize agent behavior: "fast" for performance (reduces verbosity), "interactive" for real-time conversations, and "standard" for default balanced behavior. Each mode automatically configures print and verbose settings appropriately.

```python
from swarms import Agent

# Fast mode - optimized for performance (reduces verbosity)
fast_agent = Agent(
    model_name="gpt-4o-mini",
    mode="fast",  # Disables print_on and verbose automatically
    max_loops=1
)

# Interactive mode - for real-time conversations
interactive_agent = Agent(
    model_name="gpt-4o-mini",
    mode="interactive",
    max_loops=5
)

# Standard mode - default behavior
standard_agent = Agent(
    model_name="gpt-4o-mini",
    mode="standard",
    max_loops=1
)
```

### Streaming Callback

Provide a custom callback function to receive streaming tokens in real-time. The callback is invoked for each token as it's generated, enabling custom UI updates, progress tracking, or integration with streaming interfaces. Useful for building responsive applications that display results as they're generated.

```python
from swarms import Agent

# Define a custom streaming callback
def my_streaming_callback(token: str):
    print(token, end="", flush=True)

# Agent with streaming callback
agent = Agent(
    model_name="gpt-4o-mini",
    streaming_callback=my_streaming_callback,
    max_loops=1
)

# Tokens will be streamed to the callback in real-time
response = agent.run("Tell me a story")
```

### Multiple MCP Connections

Connect to multiple MCP servers simultaneously to access tools and resources from different sources. The agent can use tools from all connected servers, enabling integration with diverse external services and APIs through a unified interface.

```python
from swarms import Agent
from swarms.schemas.mcp_schemas import MultipleMCPConnections

# Configure multiple MCP servers
mcp_configs = MultipleMCPConnections(
    connections=[
        {"server_path": "http://localhost:8000", "server_name": "server1"},
        {"server_path": "http://localhost:8001", "server_name": "server2"}
    ]
)

agent = Agent(
    model_name="gpt-4o-mini",
    mcp_configs=mcp_configs,
    max_loops=1
)

response = agent.run("Use tools from both MCP servers")
```

### Message Transforms for Context Management

Automatically manage long conversation histories by transforming messages when context limits are approached. Configure strategies like truncating oldest messages, summarizing, or chunking to maintain conversation quality while staying within token limits. Essential for long-running conversations or document processing.

```python
from swarms import Agent
from swarms.structs.transforms import TransformConfig

# Configure message transforms to handle long contexts
transforms = TransformConfig(
    max_tokens=8000,
    strategy="truncate_oldest"
)

agent = Agent(
    model_name="gpt-4o-mini",
    transforms=transforms,
    context_length=100000,
    max_loops=1
)

# Agent will automatically manage context length
response = agent.run("Process this very long conversation history")
```

### Agent with Capabilities

Define agent capabilities as metadata for better task routing and documentation. Capabilities help other agents or routing systems understand what an agent can do, enabling intelligent task delegation and agent discovery. Useful for multi-agent systems and marketplace listings.

```python
from swarms import Agent

# Agent with defined capabilities for better routing
agent = Agent(
    model_name="gpt-4o-mini",
    agent_name="Data-Analysis-Agent",
    capabilities=["data_analysis", "statistics", "visualization"],
    max_loops=1
)

response = agent.run("Analyze this dataset")
```

### Publishing to Marketplace

Publish your agent's prompt to the Swarms marketplace for sharing and reuse. When `publish_to_marketplace=True`, the agent automatically publishes its system prompt along with metadata (name, description, tags, capabilities, use cases) on initialization. Requires use_cases to be provided and SWARMS_API_KEY environment variable.

```python
from swarms import Agent

# Agent configured to publish prompt to marketplace
agent = Agent(
    model_name="gpt-4o-mini",
    agent_name="Financial-Advisor",
    agent_description="Expert financial advisor agent",
    system_prompt="You are an expert financial advisor...",
    tags=["finance", "advisor"],
    capabilities=["financial_planning", "investment_advice"],
    use_cases=[
        {"title": "Retirement Planning", "description": "Help users plan for retirement"},
        {"title": "Investment Analysis", "description": "Analyze investment opportunities"}
    ],
    publish_to_marketplace=True,
    max_loops=1
)

# Prompt will be published to marketplace on initialization
```


## New Features and Parameters

### Enhanced Run Method Parameters

The `run` method now supports several new parameters for advanced functionality:

- **`imgs`**: Process multiple images simultaneously instead of just one
- **`correct_answer`**: Validate responses against expected answers with automatic retries
- **`streaming_callback`**: Real-time token streaming for interactive applications

### MCP (Model Context Protocol) Integration

| Parameter      | Description                                         |
|----------------|-----------------------------------------------------|
| `mcp_url`      | Connect to a single MCP server                      |
| `mcp_urls`     | Connect to multiple MCP servers                     |
| `mcp_config`   | Advanced MCP configuration options for a single server |
| `mcp_configs`  | MultipleMCPConnections object for managing multiple MCP server connections |

### Advanced Reasoning and Safety

| Parameter            | Description                                                        |
|----------------------|--------------------------------------------------------------------|
| `react_on`           | Enable ReAct reasoning for complex problem-solving                 |
| `safety_prompt_on`   | Add safety constraints to agent responses                          |
| `reasoning_prompt_on`| Enable multi-loop reasoning for complex tasks                      |
| `reasoning_enabled` | Enable reasoning capabilities for supported models (e.g., o1)     |
| `reasoning_effort`   | Set reasoning effort level: "low", "medium", or "high"            |
| `thinking_tokens`   | Maximum number of thinking tokens for reasoning models            |

### Performance and Resource Management

| Parameter                | Description                                                              |
|--------------------------|--------------------------------------------------------------------------|
| `dynamic_context_window` | Automatically adjust context window based on available tokens             |
| `tool_retry_attempts`    | Configure retry behavior for tool execution                              |
| `summarize_multiple_images` | Automatically summarize results from multiple image processing         |


### Advanced Memory and Context

| Parameter                | Description                                                              |
|--------------------------|--------------------------------------------------------------------------|
| `rag_every_loop`         | Query RAG database on every loop iteration                              |
| `memory_chunk_size`      | Control memory chunk size for long-term memory                          |
| `auto_generate_prompt`   | Automatically generate system prompts based on tasks                    |
| `plan_enabled`           | Enable planning functionality for complex tasks                          |

### Artifacts and Output Management

| Parameter                | Description                                                              |
|--------------------------|--------------------------------------------------------------------------|
| `artifacts_on`           | Enable saving artifacts from agent execution                             |
| `artifacts_output_path`  | Specify where to save artifacts                                          |
| `artifacts_file_extension` | Control artifact file format                                            |

### Enhanced Tool Management

| Parameter                | Description                                                              |
|--------------------------|--------------------------------------------------------------------------|
| `tools_list_dictionary`  | Provide tool schemas in dictionary format                               |
| `tool_call_summary`      | Enable automatic summarization of tool calls                            |
| `show_tool_execution_output` | Control visibility of tool execution details                        |
| `function_calling_format_type` | Specify function calling format (OpenAI, etc.)                     |

### Advanced LLM Configuration

| Parameter                | Description                                                              |
|--------------------------|--------------------------------------------------------------------------|
| `llm_args`               | Pass additional arguments to the LLM                                     |
| `llm_base_url`           | Specify custom LLM API endpoint                                          |
| `llm_api_key`            | Provide LLM API key directly                                             |
| `top_p`                  | Control top-p sampling parameter                                         |




### Reasoning and Advanced Capabilities

| Parameter                | Description                                                              |
|--------------------------|--------------------------------------------------------------------------|
| `reasoning_enabled`      | Enable reasoning capabilities for supported models                       |
| `reasoning_effort`       | Set reasoning effort level ("low", "medium", "high")                    |
| `thinking_tokens`        | Maximum number of thinking tokens for reasoning models                   |

### Execution Modes and Marketplace

| Parameter                | Description                                                              |
|--------------------------|--------------------------------------------------------------------------|
| `mode`                   | Execution mode: "interactive", "fast", or "standard"                    |
| `capabilities`           | List of agent capabilities for documentation and routing                 |
| `publish_to_marketplace` | Publish agent prompt to Swarms marketplace                              |
| `marketplace_prompt_id`  | Load prompt from Swarms marketplace by UUID                              |

### Message Transforms and Context Management

| Parameter                | Description                                                              |
|--------------------------|--------------------------------------------------------------------------|
| `transforms`             | TransformConfig for handling context limits and message transformations |


## Best Practices

| Best Practice / Feature                                 | Description                                                                                      |
|---------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `system_prompt`                                         | Always provide a clear and concise system prompt to guide the agent's behavior.                  |
| `tools`                                                 | Use tools to extend the agent's capabilities for specific tasks.                                 |
| `retry_attempts` & error handling                       | Implement error handling and utilize the retry_attempts feature for robust execution.            |
| `long_term_memory`                                      | Leverage long_term_memory for tasks that require persistent information.                         |
| `interactive` & `dashboard`                             | Use interactive mode for real-time conversations and dashboard for monitoring.                   |
| `sentiment_analysis`                                    | Implement sentiment_analysis for applications requiring tone management.                         |
| `autosave`, `save`/`load`                               | Utilize autosave and save/load methods for continuity across sessions. Autosave saves configuration at each step to `workspace_dir/agents/{agent-name}-{uuid}/config.json`. Files created by agent tools are saved in the agent's workspace directory. |
| `dynamic_context_window` & `tokens_checks`              | Optimize token usage with dynamic_context_window and tokens_checks methods.                      |
| `concurrent` & `async` methods                          | Use concurrent and async methods for performance-critical applications.                          |
| `analyze_feedback`                                      | Regularly review and analyze feedback using the analyze_feedback method.                         |
| `artifacts_on`                                          | Use artifacts_on to save important outputs from agent execution.                                 |
| `rag_every_loop`                                        | Enable rag_every_loop when continuous context from long-term memory is needed.                   |
| `run_batched`                                           | Leverage run_batched for efficient processing of multiple related tasks.                         |
| `mcp_url` or `mcp_urls`                                 | Use mcp_url or mcp_urls to extend agent capabilities with external tools.                        |
| `react_on`                                              | Enable react_on for complex reasoning tasks requiring step-by-step analysis.                     |
| `tool_retry_attempts`                                   | Configure tool_retry_attempts for robust tool execution in production environments.              |
| `handoffs`                                              | Use handoffs to create specialized agent teams that can intelligently route tasks based on complexity and expertise requirements. |

By following these guidelines and leveraging the Swarm Agent's extensive features, you can create powerful, flexible, and efficient autonomous agents for a wide range of applications.
