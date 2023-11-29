import subprocess
import threading
import time

import pytest

from swarms.utils.code_interpreter import (
    BaseCodeInterpreter,
    SubprocessCodeInterpreter,
)


@pytest.fixture
def subprocess_code_interpreter():
    interpreter = SubprocessCodeInterpreter()
    interpreter.start_cmd = "python -c"
    yield interpreter
    interpreter.terminate()


def test_base_code_interpreter_init():
    interpreter = BaseCodeInterpreter()
    assert isinstance(interpreter, BaseCodeInterpreter)


def test_base_code_interpreter_run_not_implemented():
    interpreter = BaseCodeInterpreter()
    with pytest.raises(NotImplementedError):
        interpreter.run("code")


def test_base_code_interpreter_terminate_not_implemented():
    interpreter = BaseCodeInterpreter()
    with pytest.raises(NotImplementedError):
        interpreter.terminate()


def test_subprocess_code_interpreter_init(
    subprocess_code_interpreter,
):
    assert isinstance(
        subprocess_code_interpreter, SubprocessCodeInterpreter
    )


def test_subprocess_code_interpreter_start_process(
    subprocess_code_interpreter,
):
    subprocess_code_interpreter.start_process()
    assert subprocess_code_interpreter.process is not None


def test_subprocess_code_interpreter_terminate(
    subprocess_code_interpreter,
):
    subprocess_code_interpreter.start_process()
    subprocess_code_interpreter.terminate()
    assert subprocess_code_interpreter.process.poll() is not None


def test_subprocess_code_interpreter_run_success(
    subprocess_code_interpreter,
):
    code = 'print("Hello, World!")'
    result = list(subprocess_code_interpreter.run(code))
    assert any(
        "Hello, World!" in output.get("output", "")
        for output in result
    )


def test_subprocess_code_interpreter_run_with_error(
    subprocess_code_interpreter,
):
    code = 'print("Hello, World")\nraise ValueError("Error!")'
    result = list(subprocess_code_interpreter.run(code))
    assert any(
        "Error!" in output.get("output", "") for output in result
    )


def test_subprocess_code_interpreter_run_with_keyboard_interrupt(
    subprocess_code_interpreter,
):
    code = (
        'import time\ntime.sleep(2)\nprint("Hello, World")\nraise'
        " KeyboardInterrupt"
    )
    result = list(subprocess_code_interpreter.run(code))
    assert any(
        "KeyboardInterrupt" in output.get("output", "")
        for output in result
    )


def test_subprocess_code_interpreter_run_max_retries(
    subprocess_code_interpreter, monkeypatch
):
    def mock_subprocess_popen(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "mocked_cmd")

    monkeypatch.setattr(subprocess, "Popen", mock_subprocess_popen)

    code = 'print("Hello, World!")'
    result = list(subprocess_code_interpreter.run(code))
    assert any(
        "Maximum retries reached. Could not execute code."
        in output.get("output", "")
        for output in result
    )


def test_subprocess_code_interpreter_run_retry_on_error(
    subprocess_code_interpreter, monkeypatch
):
    def mock_subprocess_popen(*args, **kwargs):
        nonlocal popen_count
        if popen_count == 0:
            popen_count += 1
            raise subprocess.CalledProcessError(1, "mocked_cmd")
        else:
            return subprocess.Popen(
                "echo 'Hello, World!'",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

    monkeypatch.setattr(subprocess, "Popen", mock_subprocess_popen)
    popen_count = 0

    code = 'print("Hello, World!")'
    result = list(subprocess_code_interpreter.run(code))
    assert any(
        "Hello, World!" in output.get("output", "")
        for output in result
    )


# Add more tests to cover other aspects of the code and edge cases as needed

# Import statements and fixtures from the previous code block


def test_subprocess_code_interpreter_line_postprocessor(
    subprocess_code_interpreter,
):
    line = "This is a test line"
    processed_line = subprocess_code_interpreter.line_postprocessor(
        line
    )
    assert (
        processed_line == line
    )  # No processing, should remain the same


def test_subprocess_code_interpreter_preprocess_code(
    subprocess_code_interpreter,
):
    code = 'print("Hello, World!")'
    preprocessed_code = subprocess_code_interpreter.preprocess_code(
        code
    )
    assert (
        preprocessed_code == code
    )  # No preprocessing, should remain the same


def test_subprocess_code_interpreter_detect_active_line(
    subprocess_code_interpreter,
):
    line = "Active line: 5"
    active_line = subprocess_code_interpreter.detect_active_line(line)
    assert active_line == 5


def test_subprocess_code_interpreter_detect_end_of_execution(
    subprocess_code_interpreter,
):
    line = "Execution completed."
    end_of_execution = (
        subprocess_code_interpreter.detect_end_of_execution(line)
    )
    assert end_of_execution is True


def test_subprocess_code_interpreter_run_debug_mode(
    subprocess_code_interpreter, capsys
):
    subprocess_code_interpreter.debug_mode = True
    code = 'print("Hello, World!")'
    list(subprocess_code_interpreter.run(code))
    captured = capsys.readouterr()
    assert "Running code:\n" in captured.out
    assert "Received output line:\n" in captured.out


def test_subprocess_code_interpreter_run_no_debug_mode(
    subprocess_code_interpreter, capsys
):
    subprocess_code_interpreter.debug_mode = False
    code = 'print("Hello, World!")'
    list(subprocess_code_interpreter.run(code))
    captured = capsys.readouterr()
    assert "Running code:\n" not in captured.out
    assert "Received output line:\n" not in captured.out


def test_subprocess_code_interpreter_run_empty_output_queue(
    subprocess_code_interpreter,
):
    code = 'print("Hello, World!")'
    result = list(subprocess_code_interpreter.run(code))
    assert not any("active_line" in output for output in result)


def test_subprocess_code_interpreter_handle_stream_output_stdout(
    subprocess_code_interpreter,
):
    line = "This is a test line"
    subprocess_code_interpreter.handle_stream_output(
        threading.current_thread(), False
    )
    subprocess_code_interpreter.process.stdout.write(line + "\n")
    subprocess_code_interpreter.process.stdout.flush()
    time.sleep(0.1)
    output = subprocess_code_interpreter.output_queue.get()
    assert output["output"] == line


def test_subprocess_code_interpreter_handle_stream_output_stderr(
    subprocess_code_interpreter,
):
    line = "This is an error line"
    subprocess_code_interpreter.handle_stream_output(
        threading.current_thread(), True
    )
    subprocess_code_interpreter.process.stderr.write(line + "\n")
    subprocess_code_interpreter.process.stderr.flush()
    time.sleep(0.1)
    output = subprocess_code_interpreter.output_queue.get()
    assert output["output"] == line


def test_subprocess_code_interpreter_run_with_preprocess_code(
    subprocess_code_interpreter, capsys
):
    code = 'print("Hello, World!")'
    subprocess_code_interpreter.preprocess_code = (
        lambda x: x.upper()
    )  # Modify code in preprocess_code
    result = list(subprocess_code_interpreter.run(code))
    assert any(
        "Hello, World!" in output.get("output", "")
        for output in result
    )


def test_subprocess_code_interpreter_run_with_exception(
    subprocess_code_interpreter, capsys
):
    code = 'print("Hello, World!")'
    subprocess_code_interpreter.start_cmd = (  # Force an exception during subprocess creation
        "nonexistent_command"
    )
    result = list(subprocess_code_interpreter.run(code))
    assert any(
        "Maximum retries reached" in output.get("output", "")
        for output in result
    )


def test_subprocess_code_interpreter_run_with_active_line(
    subprocess_code_interpreter, capsys
):
    code = "a = 5\nprint(a)"  # Contains an active line
    result = list(subprocess_code_interpreter.run(code))
    assert any(output.get("active_line") == 5 for output in result)


def test_subprocess_code_interpreter_run_with_end_of_execution(
    subprocess_code_interpreter, capsys
):
    code = (  # Simple code without active line marker
        'print("Hello, World!")'
    )
    result = list(subprocess_code_interpreter.run(code))
    assert any(output.get("active_line") is None for output in result)


def test_subprocess_code_interpreter_run_with_multiple_lines(
    subprocess_code_interpreter, capsys
):
    code = "a = 5\nb = 10\nprint(a + b)"
    result = list(subprocess_code_interpreter.run(code))
    assert any("15" in output.get("output", "") for output in result)


def test_subprocess_code_interpreter_run_with_unicode_characters(
    subprocess_code_interpreter, capsys
):
    code = 'print("こんにちは、世界")'  # Contains unicode characters
    result = list(subprocess_code_interpreter.run(code))
    assert any(
        "こんにちは、世界" in output.get("output", "")
        for output in result
    )
