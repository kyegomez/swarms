import pytest
from swarms.structs.cron_job import (
    CronJobError,
    CronJobConfigError,
    CronJobScheduleError,
    CronJobExecutionError,
)


def test_cron_job_error_is_exception():
    """Test that CronJobError is an Exception subclass"""
    assert issubclass(CronJobError, Exception)


def test_cron_job_config_error_is_cron_job_error():
    """Test that CronJobConfigError is a CronJobError subclass"""
    assert issubclass(CronJobConfigError, CronJobError)


def test_cron_job_schedule_error_is_cron_job_error():
    """Test that CronJobScheduleError is a CronJobError subclass"""
    assert issubclass(CronJobScheduleError, CronJobError)


def test_cron_job_execution_error_is_cron_job_error():
    """Test that CronJobExecutionError is a CronJobError subclass"""
    assert issubclass(CronJobExecutionError, CronJobError)


def test_cron_job_error_can_be_raised():
    """Test that CronJobError can be raised and caught"""
    with pytest.raises(CronJobError, match="Test error"):
        raise CronJobError("Test error")


def test_cron_job_config_error_can_be_raised():
    """Test that CronJobConfigError can be raised and caught"""
    with pytest.raises(CronJobConfigError, match="Config error"):
        raise CronJobConfigError("Config error")


def test_cron_job_schedule_error_can_be_raised():
    """Test that CronJobScheduleError can be raised and caught"""
    with pytest.raises(CronJobScheduleError, match="Schedule error"):
        raise CronJobScheduleError("Schedule error")


def test_cron_job_execution_error_can_be_raised():
    """Test that CronJobExecutionError can be raised and caught"""
    with pytest.raises(CronJobExecutionError, match="Execution error"):
        raise CronJobExecutionError("Execution error")


def test_cron_job_error_inheritance_chain():
    """Test that all CronJob errors inherit from CronJobError and Exception"""
    assert issubclass(CronJobConfigError, Exception)
    assert issubclass(CronJobScheduleError, Exception)
    assert issubclass(CronJobExecutionError, Exception)


def test_cron_job_error_with_custom_message():
    """Test CronJobError with custom message"""
    error = CronJobError("Custom error message")
    assert str(error) == "Custom error message"


def test_cron_job_config_error_with_custom_message():
    """Test CronJobConfigError with custom message"""
    error = CronJobConfigError("Invalid configuration")
    assert str(error) == "Invalid configuration"


def test_cron_job_schedule_error_with_custom_message():
    """Test CronJobScheduleError with custom message"""
    error = CronJobScheduleError("Scheduling failed")
    assert str(error) == "Scheduling failed"


def test_cron_job_execution_error_with_custom_message():
    """Test CronJobExecutionError with custom message"""
    error = CronJobExecutionError("Task failed")
    assert str(error) == "Task failed"
