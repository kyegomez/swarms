def check_done(s):
    return "<DONE>" in s


def check_finished(s):
    return "finished" in s


def check_complete(s):
    return "complete" in s


def check_success(s):
    return "success" in s


def check_failure(s):
    return "failure" in s


def check_error(s):
    return "error" in s


def check_stopped(s):
    return "stopped" in s


def check_cancelled(s):
    return "cancelled" in s


def check_exit(s):
    return "exit" in s


def check_end(s):
    return "end" in s


def check_stopping_conditions(input: str) -> str:
    """
    Checks a string against all stopping conditions and returns an appropriate message.

    Args:
        s (str): The input string to check

    Returns:
        str: A message indicating which stopping condition was met, or None if no condition was met
    """
    conditions = [
        (check_done, "Task is done"),
        (check_finished, "Task is finished"),
        (check_complete, "Task is complete"),
        (check_success, "Task succeeded"),
        (check_failure, "Task failed"),
        (check_error, "Task encountered an error"),
        (check_stopped, "Task was stopped"),
        (check_cancelled, "Task was cancelled"),
        (check_exit, "Task exited"),
        (check_end, "Task ended"),
    ]

    for check_func, message in conditions:
        if check_func(input):
            return message

    return None
