# Best LLM Models by Task Type
# Simplified dictionary structure with model names and categories

best_models = {
    "Vision": [
        {"model": "gemini/gemini-2.5-pro", "category": "Vision"},
    ],
    "text-generation": [
        {
            "model": "claude-sonnet-4-20250514",
            "category": "text-generation",
        },
        {"model": "gpt-5-chat", "category": "text-generation"},
    ],
}


# Function to get all models for a task type
def get_models_by_task(task_type: str) -> list:
    """
    Get all models for a specific task type.

    Args:
        task_type (str): The task category (e.g., 'WebDev', 'Vision', 'text-generation')

    Returns:
        list: List of all models for the task type
    """
    if task_type not in best_models:
        raise ValueError(
            f"Task type '{task_type}' not found. Available types: {list(best_models.keys())}"
        )

    return best_models[task_type]


# Function to get the first model for a task type (simplified from get_top_model)
def get_first_model(task_type: str) -> dict:
    """
    Get the first model for a specific task type.

    Args:
        task_type (str): The task category (e.g., 'WebDev', 'Vision', 'text-generation')

    Returns:
        dict: First model information with model name and category
    """
    if task_type not in best_models:
        raise ValueError(
            f"Task type '{task_type}' not found. Available types: {list(best_models.keys())}"
        )

    models = best_models[task_type]
    if not models:
        raise ValueError(
            f"No models found for task type '{task_type}'"
        )

    return models[0]


# Function to search for a specific model across all categories
def find_model_by_name(model_name: str) -> dict:
    """
    Find a model by name across all task categories.

    Args:
        model_name (str): The model name to search for

    Returns:
        dict: Model information if found, None otherwise
    """
    for task_type, models in best_models.items():
        for model in models:
            if model["model"].lower() == model_name.lower():
                return model
    return None


# Function to get all available task types
def get_available_task_types() -> list:
    """
    Get all available task types/categories.

    Returns:
        list: List of all task type names
    """
    return list(best_models.keys())
