import asyncio

from loguru import logger

from swarms.utils.litellm_wrapper import LiteLLM

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Retrieve detailed current weather information for a specified location, including temperature, humidity, wind speed, and atmospheric conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA, or a specific geographic coordinate in the format 'latitude,longitude'.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit", "kelvin"],
                        "description": "The unit of temperature measurement to be used in the response.",
                    },
                    "include_forecast": {
                        "type": "boolean",
                        "description": "Indicates whether to include a short-term weather forecast along with the current conditions.",
                    },
                    "time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Optional parameter to specify the time for which the weather data is requested, in ISO 8601 format.",
                    },
                },
                "required": [
                    "location",
                    "unit",
                    "include_forecast",
                    "time",
                ],
            },
        },
    }
]

# Initialize LiteLLM with streaming enabled
llm = LiteLLM(model_name="gpt-4o-mini", tools_list_dictionary=tools)


async def main():
    # When streaming is enabled, arun returns a stream of chunks
    # We need to handle these chunks directly, not try to access .choices
    stream = await llm.arun("What is the weather in San Francisco?")

    logger.info(f"Received stream from LLM. {stream}")

    if stream is not None:
        logger.info(f"Stream is not None. {stream}")
    else:
        logger.info("Stream is None.")


def run_test_suite():
    """Run all test cases and generate a comprehensive report."""
    logger.info("Starting LiteLLM Test Suite")
    total_tests = 0
    passed_tests = 0
    failed_tests = []

    def log_test_result(test_name: str, passed: bool, error=None):
        nonlocal total_tests, passed_tests
        total_tests += 1
        if passed:
            passed_tests += 1
            logger.success(f"✅ {test_name} - PASSED")
        else:
            failed_tests.append((test_name, error))
            logger.error(f"❌ {test_name} - FAILED: {error}")

    # Test 1: Basic Initialization
    try:
        logger.info("Testing basic initialization")
        llm = LiteLLM()
        assert llm.model_name == "gpt-4.1"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 4000
        log_test_result("Basic Initialization", True)
    except Exception as e:
        log_test_result("Basic Initialization", False, str(e))

    # Test 2: Custom Parameters
    try:
        logger.info("Testing custom parameters")
        llm = LiteLLM(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2000,
            system_prompt="You are a helpful assistant",
        )
        assert llm.model_name == "gpt-3.5-turbo"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 2000
        assert llm.system_prompt == "You are a helpful assistant"
        log_test_result("Custom Parameters", True)
    except Exception as e:
        log_test_result("Custom Parameters", False, str(e))

    # Test 3: Message Preparation
    try:
        logger.info("Testing message preparation")
        llm = LiteLLM(system_prompt="Test system prompt")
        messages = llm._prepare_messages("Test task")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Test system prompt"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test task"
        log_test_result("Message Preparation", True)
    except Exception as e:
        log_test_result("Message Preparation", False, str(e))

    # Test 4: Basic Completion
    try:
        logger.info("Testing basic completion")
        llm = LiteLLM()
        response = llm.run("What is 2+2?")
        assert isinstance(response, str)
        assert len(response) > 0
        log_test_result("Basic Completion", True)
    except Exception as e:
        log_test_result("Basic Completion", False, str(e))

    try:
        # tool usage
        asyncio.run(main())
    except Exception as e:
        log_test_result("Tool Usage", False, str(e))

    # Test 5: Tool Calling
    try:
        logger.info("Testing tool calling")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_function",
                    "description": "A test function",
                    "parameters": {
                        "type": "object",
                        "properties": {"test": {"type": "string"}},
                    },
                },
            }
        ]
        llm = LiteLLM(
            tools_list_dictionary=tools,
            tool_choice="auto",
            model_name="gpt-4o-mini",
        )
        assert llm.tools_list_dictionary == tools
        assert llm.tool_choice == "auto"
        log_test_result("Tool Calling Setup", True)
    except Exception as e:
        log_test_result("Tool Calling Setup", False, str(e))

    # Test 6: Async Completion
    async def test_async():
        try:
            logger.info("Testing async completion")
            llm = LiteLLM()
            response = await llm.arun("What is 3+3?")
            assert isinstance(response, str)
            assert len(response) > 0
            log_test_result("Async Completion", True)
        except Exception as e:
            log_test_result("Async Completion", False, str(e))

    asyncio.run(test_async())

    # Test 7: Batched Run
    try:
        logger.info("Testing batched run")
        llm = LiteLLM()
        tasks = ["Task 1", "Task 2", "Task 3"]
        responses = llm.batched_run(tasks, batch_size=2)
        assert isinstance(responses, list)
        assert len(responses) == 3
        log_test_result("Batched Run", True)
    except Exception as e:
        log_test_result("Batched Run", False, str(e))

    # Test 8: Vision Support Check
    try:
        logger.info("Testing vision support check")
        llm = LiteLLM(model_name="gpt-4.1")
        # This should not raise an error for vision-capable models
        llm.check_if_model_supports_vision(img="test.jpg")
        log_test_result("Vision Support Check", True)
    except Exception as e:
        log_test_result("Vision Support Check", False, str(e))

    # Test 9: Direct URL Processing
    try:
        logger.info("Testing direct URL processing")
        llm = LiteLLM(model_name="gpt-4.1")
        test_url = "https://github.com/kyegomez/swarms/blob/master/swarms_logo_new.png?raw=true"
        should_use_direct = llm._should_use_direct_url(test_url)
        assert isinstance(should_use_direct, bool)
        log_test_result("Direct URL Processing", True)
    except Exception as e:
        log_test_result("Direct URL Processing", False, str(e))

    # Test 10: Message Preparation with Image
    try:
        logger.info("Testing message preparation with image")
        llm = LiteLLM(model_name="gpt-4.1")
        # Mock image URL to test message structure
        test_img = "https://github.com/kyegomez/swarms/blob/master/swarms_logo_new.png?raw=true"
        messages = llm._prepare_messages(
            "Describe this image", img=test_img
        )
        assert isinstance(messages, list)
        assert len(messages) >= 1
        # Check if image content is properly structured
        user_message = next(
            (msg for msg in messages if msg["role"] == "user"), None
        )
        assert user_message is not None
        log_test_result("Message Preparation with Image", True)
    except Exception as e:
        log_test_result(
            "Message Preparation with Image", False, str(e)
        )

    # Test 11: Vision Processing Methods
    try:
        logger.info("Testing vision processing methods")
        llm = LiteLLM(model_name="gpt-4.1")
        messages = []

        # Test OpenAI vision processing
        processed_messages = llm.openai_vision_processing(
            "Describe this image",
            "https://github.com/kyegomez/swarms/blob/master/swarms_logo_new.png?raw=true",
            messages.copy(),
        )
        assert isinstance(processed_messages, list)
        assert len(processed_messages) > 0

        # Test Anthropic vision processing
        llm_anthropic = LiteLLM(
            model_name="claude-3-5-sonnet-20241022"
        )
        processed_messages_anthropic = llm_anthropic.anthropic_vision_processing(
            "Describe this image",
            "https://github.com/kyegomez/swarms/blob/master/swarms_logo_new.png?raw=true",
            messages.copy(),
        )
        assert isinstance(processed_messages_anthropic, list)
        assert len(processed_messages_anthropic) > 0

        log_test_result("Vision Processing Methods", True)
    except Exception as e:
        log_test_result("Vision Processing Methods", False, str(e))

    # Test 12: Local vs URL Detection
    try:
        logger.info("Testing local vs URL detection")
        llm = LiteLLM(model_name="gpt-4.1")

        # Test URL detection
        url_test = "https://github.com/kyegomez/swarms/blob/master/swarms_logo_new.png?raw=true"
        is_url_direct = llm._should_use_direct_url(url_test)

        # Test local file detection
        local_test = "/path/to/local/image.jpg"
        is_local_direct = llm._should_use_direct_url(local_test)

        # URLs should potentially use direct, local files should not
        assert isinstance(is_url_direct, bool)
        assert isinstance(is_local_direct, bool)
        assert (
            is_local_direct is False
        )  # Local files should never use direct URL

        log_test_result("Local vs URL Detection", True)
    except Exception as e:
        log_test_result("Local vs URL Detection", False, str(e))

    # Test 13: Vision Message Structure
    try:
        logger.info("Testing vision message structure")
        llm = LiteLLM(model_name="gpt-4.1")
        messages = []

        # Test message structure for image input
        result = llm.vision_processing(
            task="What do you see?",
            image="https://github.com/kyegomez/swarms/blob/master/swarms_logo_new.png?raw=true",
            messages=messages,
        )

        assert isinstance(result, list)
        assert len(result) > 0

        # Verify the message contains both text and image components
        user_msg = result[-1]  # Last message should be user message
        assert user_msg["role"] == "user"
        assert "content" in user_msg

        log_test_result("Vision Message Structure", True)
    except Exception as e:
        log_test_result("Vision Message Structure", False, str(e))

    # Generate test report
    success_rate = (passed_tests / total_tests) * 100
    logger.info("\n=== Test Suite Report ===")
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed Tests: {passed_tests}")
    logger.info(f"Failed Tests: {len(failed_tests)}")
    logger.info(f"Success Rate: {success_rate:.2f}%")

    if failed_tests:
        logger.error("\nFailed Tests Details:")
        for test_name, error in failed_tests:
            logger.error(f"{test_name}: {error}")

    return {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "success_rate": success_rate,
    }


if __name__ == "__main__":
    test_results = run_test_suite()
    logger.info(
        "Test suite completed. Check test_litellm.log for detailed logs."
    )
