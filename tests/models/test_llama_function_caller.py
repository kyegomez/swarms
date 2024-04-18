import pytest

from swarms.models.llama_function_caller import LlamaFunctionCaller


# Define fixtures if needed
@pytest.fixture
def llama_caller():
    # Initialize the LlamaFunctionCaller with a sample model
    return LlamaFunctionCaller()


# Basic test for model loading
def test_llama_model_loading(llama_caller):
    assert llama_caller.model is not None
    assert llama_caller.tokenizer is not None


# Test adding and calling custom functions
def test_llama_custom_function(llama_caller):
    def sample_function(arg1, arg2):
        return f"Sample function called with args: {arg1}, {arg2}"

    llama_caller.add_func(
        name="sample_function",
        function=sample_function,
        description="Sample custom function",
        arguments=[
            {
                "name": "arg1",
                "type": "string",
                "description": "Argument 1",
            },
            {
                "name": "arg2",
                "type": "string",
                "description": "Argument 2",
            },
        ],
    )

    result = llama_caller.call_function(
        "sample_function", arg1="arg1_value", arg2="arg2_value"
    )
    assert (
        result
        == "Sample function called with args: arg1_value, arg2_value"
    )


# Test streaming user prompts
def test_llama_streaming(llama_caller):
    user_prompt = "Tell me about the tallest mountain in the world."
    response = llama_caller(user_prompt)
    assert isinstance(response, str)
    assert len(response) > 0


# Test custom function not found
def test_llama_custom_function_not_found(llama_caller):
    with pytest.raises(ValueError):
        llama_caller.call_function("non_existent_function")


# Test invalid arguments for custom function
def test_llama_custom_function_invalid_arguments(llama_caller):
    def sample_function(arg1, arg2):
        return f"Sample function called with args: {arg1}, {arg2}"

    llama_caller.add_func(
        name="sample_function",
        function=sample_function,
        description="Sample custom function",
        arguments=[
            {
                "name": "arg1",
                "type": "string",
                "description": "Argument 1",
            },
            {
                "name": "arg2",
                "type": "string",
                "description": "Argument 2",
            },
        ],
    )

    with pytest.raises(TypeError):
        llama_caller.call_function("sample_function", arg1="arg1_value")


# Test streaming with custom runtime
def test_llama_custom_runtime():
    llama_caller = LlamaFunctionCaller(
        model_id="Your-Model-ID",
        cache_dir="Your-Cache-Directory",
        runtime="cuda",
    )
    user_prompt = "Tell me about the tallest mountain in the world."
    response = llama_caller(user_prompt)
    assert isinstance(response, str)
    assert len(response) > 0


# Test caching functionality
def test_llama_cache():
    llama_caller = LlamaFunctionCaller(
        model_id="Your-Model-ID",
        cache_dir="Your-Cache-Directory",
        runtime="cuda",
    )

    # Perform a request to populate the cache
    user_prompt = "Tell me about the tallest mountain in the world."
    response = llama_caller(user_prompt)

    # Check if the response is retrieved from the cache
    llama_caller.model.from_cache = True
    response_from_cache = llama_caller(user_prompt)
    assert response == response_from_cache


# Test response length within max_tokens limit
def test_llama_response_length():
    llama_caller = LlamaFunctionCaller(
        model_id="Your-Model-ID",
        cache_dir="Your-Cache-Directory",
        runtime="cuda",
    )

    # Generate a long prompt
    long_prompt = "A " + "test " * 100  # Approximately 500 tokens

    # Ensure the response does not exceed max_tokens
    response = llama_caller(long_prompt)
    assert len(response.split()) <= 500


# Add more test cases as needed to cover different aspects of your code

# ...
