import time
from swarms.utils.llm_metrcs_decorator import metrics_decorator


def test_metrics_decorator():
    @metrics_decorator
    def test_func():
        time.sleep(0.1)  # simulate some work
        return list(range(100))  # return a list of 100 tokens

    result = test_func()
    lines = result.strip().split("\n")

    # Check that the decorator returns 3 lines of output
    assert len(lines) == 3

    # Check that the Time to First Token is less than or equal to the Generation Latency
    time_to_first_token = float(lines[0].split(": ")[1])
    generation_latency = float(lines[1].split(": ")[1])
    assert time_to_first_token <= generation_latency

    # Check that the Throughput is approximately equal to the number of tokens divided by the Generation Latency
    throughput = float(lines[2].split(": ")[1])
    assert (
        abs(throughput - 100 / generation_latency) < 0.01
    )  # allow for a small amount of error


def test_metrics_decorator_1_token():
    @metrics_decorator
    def test_func():
        time.sleep(0.1)  # simulate some work
        return [0]  # return a list of 1 token

    result = test_func()
    lines = result.strip().split("\n")
    assert len(lines) == 3
    time_to_first_token = float(lines[0].split(": ")[1])
    generation_latency = float(lines[1].split(": ")[1])
    assert time_to_first_token <= generation_latency
    throughput = float(lines[2].split(": ")[1])
    assert abs(throughput - 1 / generation_latency) < 0.01


# Repeat the test with different numbers of tokens and different amounts of work
for i in range(2, 17):

    def test_func():
        @metrics_decorator
        def test_func():
            time.sleep(0.01 * i)  # simulate some work
            return list(range(i))  # return a list of i tokens

        result = test_func()
        lines = result.strip().split("\n")
        assert len(lines) == 3
        time_to_first_token = float(lines[0].split(": ")[1])
        generation_latency = float(lines[1].split(": ")[1])
        assert time_to_first_token <= generation_latency
        throughput = float(lines[2].split(": ")[1])
        assert abs(throughput - i / generation_latency) < 0.01

    globals()[f"test_metrics_decorator_{i}_tokens"] = test_func
