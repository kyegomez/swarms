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
