import pytest
from swarms.utils.math_eval import math_eval


def test_math_eval_same_output():
    @math_eval(lambda x: x + 1, lambda x: x + 1)
    def func(x):
        return x

    for i in range(20):
        result1, result2 = func(i)
        assert result1 == result2
        assert result1 == i + 1


def test_math_eval_different_output():
    @math_eval(lambda x: x + 1, lambda x: x + 2)
    def func(x):
        return x

    for i in range(20):
        result1, result2 = func(i)
        assert result1 != result2
        assert result1 == i + 1
        assert result2 == i + 2


def test_math_eval_exception_in_func1():
    @math_eval(lambda x: 1 / x, lambda x: x)
    def func(x):
        return x

    with pytest.raises(ZeroDivisionError):
        func(0)


def test_math_eval_exception_in_func2():
    @math_eval(lambda x: x, lambda x: 1 / x)
    def func(x):
        return x

    with pytest.raises(ZeroDivisionError):
        func(0)


def test_math_eval_with_multiple_arguments():
    @math_eval(lambda x, y: x + y, lambda x, y: y + x)
    def func(x, y):
        return x, y

    for i in range(10):
        for j in range(10):
            result1, result2 = func(i, j)
            assert result1 == result2
            assert result1 == i + j


def test_math_eval_with_kwargs():
    @math_eval(lambda x, y=0: x + y, lambda x, y=0: y + x)
    def func(x, y=0):
        return x, y

    for i in range(10):
        for j in range(10):
            result1, result2 = func(i, y=j)
            assert result1 == result2
            assert result1 == i + j


def test_math_eval_with_no_arguments():
    @math_eval(lambda: 1, lambda: 1)
    def func():
        return

    result1, result2 = func()
    assert result1 == result2
    assert result1 == 1


def test_math_eval_with_different_types():
    @math_eval(lambda x: str(x), lambda x: x)
    def func(x):
        return x

    for i in range(10):
        result1, result2 = func(i)
        assert result1 != result2
        assert result1 == str(i)
        assert result2 == i
