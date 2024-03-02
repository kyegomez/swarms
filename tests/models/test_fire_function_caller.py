from unittest.mock import MagicMock

from swarms.models.fire_function import FireFunctionCaller


def test_fire_function_caller_run(mocker):
    # Create mock model and tokenizer
    model = MagicMock()
    tokenizer = MagicMock()
    mocker.patch.object(FireFunctionCaller, "model", model)
    mocker.patch.object(FireFunctionCaller, "tokenizer", tokenizer)

    # Create mock task and arguments
    task = "Add 2 and 3"
    args = (2, 3)
    kwargs = {}

    # Create mock generated_ids and decoded output
    generated_ids = [1, 2, 3]
    decoded_output = "5"
    model.generate.return_value = generated_ids
    tokenizer.batch_decode.return_value = [decoded_output]

    # Create FireFunctionCaller instance
    fire_function_caller = FireFunctionCaller()

    # Run the function
    fire_function_caller.run(task, *args, **kwargs)

    # Assert model.generate was called with the correct inputs
    model.generate.assert_called_once_with(
        tokenizer.apply_chat_template.return_value,
        max_new_tokens=fire_function_caller.max_tokens,
        *args,
        **kwargs,
    )

    # Assert tokenizer.batch_decode was called with the correct inputs
    tokenizer.batch_decode.assert_called_once_with(generated_ids)

    # Assert the decoded output is printed
    assert decoded_output in mocker.patch.object(print, "call_args_list")
