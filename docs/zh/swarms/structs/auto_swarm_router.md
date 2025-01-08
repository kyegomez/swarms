# AutoSwarmRouter

The `AutoSwarmRouter` class is designed to route tasks to the appropriate swarm based on the provided name. This class allows for customization of preprocessing, routing, and postprocessing of tasks, making it highly adaptable to various workflows and requirements.

### Key Concepts

- **Routing**: Directing tasks to the appropriate swarm based on specific criteria.
- **Preprocessing and Postprocessing**: Customizable functions to handle tasks before and after routing.
- **Swarms**: Collections of `BaseSwarm` objects that perform the tasks.

## Attributes

### Arguments

| Argument           | Type                             | Default   | Description |
|--------------------|----------------------------------|-----------|-------------|
| `name`             | `Optional[str]`                  | `None`    | The name of the router. |
| `description`      | `Optional[str]`                  | `None`    | The description of the router. |
| `verbose`          | `bool`                           | `False`   | Whether to enable verbose mode. |
| `custom_params`    | `Optional[Dict[str, Any]]`       | `None`    | Custom parameters for the router. |
| `swarms`           | `Sequence[BaseSwarm]`            | `None`    | A list of `BaseSwarm` objects. |
| `custom_preprocess`| `Optional[Callable]`             | `None`    | Custom preprocessing function for tasks. |
| `custom_postprocess`| `Optional[Callable]`            | `None`    | Custom postprocessing function for task results. |
| `custom_router`    | `Optional[Callable]`             | `None`    | Custom routing function for tasks. |

### Attributes

| Attribute            | Type                             | Description |
|----------------------|----------------------------------|-------------|
| `name`               | `Optional[str]`                  | The name of the router. |
| `description`        | `Optional[str]`                  | The description of the router. |
| `verbose`            | `bool`                           | Whether to enable verbose mode. |
| `custom_params`      | `Optional[Dict[str, Any]]`       | Custom parameters for the router. |
| `swarms`             | `Sequence[BaseSwarm]`            | A list of `BaseSwarm` objects. |
| `custom_preprocess`  | `Optional[Callable]`             | Custom preprocessing function for tasks. |
| `custom_postprocess` | `Optional[Callable]`             | Custom postprocessing function for task results. |
| `custom_router`      | `Optional[Callable]`             | Custom routing function for tasks. |
| `swarm_dict`         | `Dict[str, BaseSwarm]`           | A dictionary of swarms keyed by their name. |

## Methods

### run

Executes the swarm simulation and routes the task to the appropriate swarm.

**Arguments:**

| Parameter | Type    | Default | Description |
|-----------|---------|---------|-------------|
| `task`    | `str`   | `None`  | The task to be executed. |
| `*args`   |         |         | Additional arguments. |
| `**kwargs`|         |         | Additional keyword arguments. |

**Returns:**

| Return Type | Description |
|-------------|-------------|
| `Any`       | The result of the routed task. |

**Raises:**

- `ValueError`: If the specified swarm is not found.
- `Exception`: If any error occurs during task routing or execution.

**Examples:**

```python
router = AutoSwarmRouter(name="example_router", swarms=[swarm1, swarm2])

# Running a task
result = router.run(task="example_task")
```

### len_of_swarms

Prints the number of swarms available in the router.

**Examples:**

```python
router = AutoSwarmRouter(name="example_router", swarms=[swarm1, swarm2])

# Printing the number of swarms
router.len_of_swarms()  # Output: 2
```

### list_available_swarms

Logs the available swarms and their descriptions.

**Examples:**

```python
router = AutoSwarmRouter(name="example_router", swarms=[swarm1, swarm2])

# Listing available swarms
router.list_available_swarms()
# Output:
# INFO: Swarm Name: swarm1 || Swarm Description: Description of swarm1
# INFO: Swarm Name: swarm2 || Swarm Description: Description of swarm2
```

### Additional Examples

#### Example 1: Custom Preprocessing and Postprocessing

```python
def custom_preprocess(task, *args, **kwargs):
    # Custom preprocessing logic
    task = task.upper()
    return task, args, kwargs

def custom_postprocess(result):
    # Custom postprocessing logic
    return result.lower()

router = AutoSwarmRouter(
    name="example_router",
    swarms=[swarm1, swarm2],
    custom_preprocess=custom_preprocess,
    custom_postprocess=custom_postprocess
)

# Running a task with custom preprocessing and postprocessing
result = router.run(task="example_task")
print(result)  # Output will be the processed result
```

#### Example 2: Custom Router Function

```python
def custom_router(router, task, *args, **kwargs):
    # Custom routing logic
    if "specific" in task:
        return router.swarm_dict["specific_swarm"].run(task, *args, **kwargs)
    return router.swarm_dict["default_swarm"].run(task, *args, **kwargs)

router = AutoSwarmRouter(
    name="example_router",
    swarms=[default_swarm, specific_swarm],
    custom_router=custom_router
)

# Running a task with custom routing
result = router.run(task="specific_task")
print(result)  # Output will be the result of the routed task
```

#### Example 3: Verbose Mode

```python
router = AutoSwarmRouter(
    name="example_router",
    swarms=[swarm1, swarm2],
    verbose=True
)

# Running a task with verbose mode enabled
result = router.run(task="example_task")
# Output will include detailed logs of the task routing and execution process
```

## Summary

The `AutoSwarmRouter` class provides a flexible and customizable approach to routing tasks to appropriate swarms, supporting custom preprocessing, routing, and postprocessing functions. This makes it a powerful tool for managing complex workflows that require dynamic task handling and execution.