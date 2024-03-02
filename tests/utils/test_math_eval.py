from swarms.utils import math_eval


def func1_no_exception(x):
    return x + 2


def func2_no_exception(x):
    return x + 2


def func1_with_exception(x):
    raise ValueError()


def func2_with_exception(x):
    raise ValueError()


def test_same_results_no_exception(caplog):
    @math_eval(func1_no_exception, func2_no_exception)
    def test_func(x):
        return x

    result1, result2 = test_func(5)
    assert result1 == result2 == 7
    assert "Outputs do not match" not in caplog.text


def test_func1_exception(caplog):
    @math_eval(func1_with_exception, func2_no_exception)
    def test_func(x):
        return x

    result1, result2 = test_func(5)
    assert result1 is None
    assert result2 == 7
    assert "Error in func1:" in caplog.text


# similar tests for func2_with_exception and when func1 and func2 return different results
