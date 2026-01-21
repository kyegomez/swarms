import pytest
from swarms.structs.stopping_conditions import (
    check_done,
    check_finished,
    check_complete,
    check_success,
    check_failure,
    check_error,
    check_stopped,
    check_cancelled,
    check_exit,
    check_end,
    check_stopping_conditions,
)


def test_check_done_true():
    """Test check_done returns True when <DONE> is in string"""
    assert check_done("Task is <DONE>") is True


def test_check_done_false():
    """Test check_done returns False when <DONE> is not in string"""
    assert check_done("Task in progress") is False


def test_check_finished_true():
    """Test check_finished returns True when 'finished' is in string"""
    assert check_finished("Task finished successfully") is True


def test_check_finished_false():
    """Test check_finished returns False when 'finished' is not in string"""
    assert check_finished("Task in progress") is False


def test_check_complete_true():
    """Test check_complete returns True when 'complete' is in string"""
    assert check_complete("Task is complete") is True


def test_check_complete_false():
    """Test check_complete returns False when 'complete' is not in string"""
    assert check_complete("Task in progress") is False


def test_check_success_true():
    """Test check_success returns True when 'success' is in string"""
    assert check_success("Task success") is True


def test_check_success_false():
    """Test check_success returns False when 'success' is not in string"""
    assert check_success("Task failed") is False


def test_check_failure_true():
    """Test check_failure returns True when 'failure' is in string"""
    assert check_failure("Task failure detected") is True


def test_check_failure_false():
    """Test check_failure returns False when 'failure' is not in string"""
    assert check_failure("Task succeeded") is False


def test_check_error_true():
    """Test check_error returns True when 'error' is in string"""
    assert check_error("An error occurred") is True


def test_check_error_false():
    """Test check_error returns False when 'error' is not in string"""
    assert check_error("Everything is fine") is False


def test_check_stopped_true():
    """Test check_stopped returns True when 'stopped' is in string"""
    assert check_stopped("Task was stopped") is True


def test_check_stopped_false():
    """Test check_stopped returns False when 'stopped' is not in string"""
    assert check_stopped("Task is running") is False


def test_check_cancelled_true():
    """Test check_cancelled returns True when 'cancelled' is in string"""
    assert check_cancelled("Task cancelled by user") is True


def test_check_cancelled_false():
    """Test check_cancelled returns False when 'cancelled' is not in string"""
    assert check_cancelled("Task is running") is False


def test_check_exit_true():
    """Test check_exit returns True when 'exit' is in string"""
    assert check_exit("Program will exit") is True


def test_check_exit_false():
    """Test check_exit returns False when 'exit' is not in string"""
    assert check_exit("Task is running") is False


def test_check_end_true():
    """Test check_end returns True when 'end' is in string"""
    assert check_end("This is the end") is True


def test_check_end_false():
    """Test check_end returns False when 'end' is not in string"""
    assert check_end("Task is running") is False


def test_check_stopping_conditions_done():
    """Test check_stopping_conditions returns correct message for done"""
    result = check_stopping_conditions("Task is <DONE>")
    assert result == "Task is done"


def test_check_stopping_conditions_finished():
    """Test check_stopping_conditions returns correct message for finished"""
    result = check_stopping_conditions("Task finished successfully")
    assert result == "Task is finished"


def test_check_stopping_conditions_complete():
    """Test check_stopping_conditions returns correct message for complete"""
    result = check_stopping_conditions("Task is complete")
    assert result == "Task is complete"


def test_check_stopping_conditions_success():
    """Test check_stopping_conditions returns correct message for success"""
    result = check_stopping_conditions("Task success")
    assert result == "Task succeeded"


def test_check_stopping_conditions_failure():
    """Test check_stopping_conditions returns correct message for failure"""
    result = check_stopping_conditions("Task failure")
    assert result == "Task failed"


def test_check_stopping_conditions_error():
    """Test check_stopping_conditions returns correct message for error"""
    result = check_stopping_conditions("An error occurred")
    assert result == "Task encountered an error"


def test_check_stopping_conditions_stopped():
    """Test check_stopping_conditions returns correct message for stopped"""
    result = check_stopping_conditions("Task was stopped")
    assert result == "Task was stopped"


def test_check_stopping_conditions_cancelled():
    """Test check_stopping_conditions returns correct message for cancelled"""
    result = check_stopping_conditions("Task was cancelled")
    assert result == "Task was cancelled"


def test_check_stopping_conditions_exit():
    """Test check_stopping_conditions returns correct message for exit"""
    result = check_stopping_conditions("Program will exit")
    assert result == "Task exited"


def test_check_stopping_conditions_end():
    """Test check_stopping_conditions returns correct message for end"""
    result = check_stopping_conditions("This is the end")
    assert result == "Task ended"


def test_check_stopping_conditions_none():
    """Test check_stopping_conditions returns None when no condition is met"""
    result = check_stopping_conditions("Task is running normally")
    assert result is None


def test_check_stopping_conditions_priority():
    """Test that check_stopping_conditions returns first matching condition"""
    # 'finished' appears before other conditions in the list, so it should match first
    result = check_stopping_conditions("Task is finished and complete")
    assert result == "Task is finished"


def test_check_stopping_conditions_case_sensitive():
    """Test that all checks are case-sensitive"""
    assert check_done("DONE") is False  # Should be <DONE>
    assert check_finished("FINISHED") is False  # lowercase 'finished' required
    assert check_complete("COMPLETE") is False  # lowercase 'complete' required
