import pytest
from swarms.models.vllm import vLLM


# Fixture for initializing vLLM
@pytest.fixture
def vllm_instance():
    return vLLM()


# Test the default initialization of vLLM
def test_vllm_default_init(vllm_instance):
    assert isinstance(vllm_instance, vLLM)
    assert vllm_instance.model_name == "facebook/opt-13b"
    assert vllm_instance.tensor_parallel_size == 4
    assert not vllm_instance.trust_remote_code
    assert vllm_instance.revision is None
    assert vllm_instance.temperature == 0.5
    assert vllm_instance.top_p == 0.95


# Test custom initialization of vLLM
def test_vllm_custom_init():
    vllm_instance = vLLM(
        model_name="custom_model",
        tensor_parallel_size=8,
        trust_remote_code=True,
        revision="123",
        temperature=0.7,
        top_p=0.9,
    )
    assert isinstance(vllm_instance, vLLM)
    assert vllm_instance.model_name == "custom_model"
    assert vllm_instance.tensor_parallel_size == 8
    assert vllm_instance.trust_remote_code
    assert vllm_instance.revision == "123"
    assert vllm_instance.temperature == 0.7
    assert vllm_instance.top_p == 0.9


# Test the run method of vLLM
def test_vllm_run(vllm_instance):
    task = "Hello, vLLM!"
    result = vllm_instance.run(task)
    assert isinstance(result, str)
    assert len(result) > 0


# Test run method with different temperature and top_p values
@pytest.mark.parametrize(
    "temperature, top_p", [(0.2, 0.8), (0.8, 0.2)]
)
def test_vllm_run_with_params(vllm_instance, temperature, top_p):
    task = "Temperature and Top-P Test"
    result = vllm_instance.run(
        task, temperature=temperature, top_p=top_p
    )
    assert isinstance(result, str)
    assert len(result) > 0


# Test run method with a specific model revision
def test_vllm_run_with_revision(vllm_instance):
    task = "Specific Model Revision Test"
    result = vllm_instance.run(task, revision="abc123")
    assert isinstance(result, str)
    assert len(result) > 0


# Test run method with a specific model name
def test_vllm_run_with_custom_model(vllm_instance):
    task = "Custom Model Test"
    custom_model_name = "my_custom_model"
    result = vllm_instance.run(task, model_name=custom_model_name)
    assert isinstance(result, str)
    assert len(result) > 0
    assert vllm_instance.model_name == custom_model_name


# Test run method with invalid task input
def test_vllm_run_invalid_task(vllm_instance):
    invalid_task = None
    with pytest.raises(ValueError):
        vllm_instance.run(invalid_task)


# Test run method with a very high temperature value
def test_vllm_run_high_temperature(vllm_instance):
    task = "High Temperature Test"
    high_temperature = 10.0
    result = vllm_instance.run(task, temperature=high_temperature)
    assert isinstance(result, str)
    assert len(result) > 0


# Test run method with a very low top_p value
def test_vllm_run_low_top_p(vllm_instance):
    task = "Low Top-P Test"
    low_top_p = 0.01
    result = vllm_instance.run(task, top_p=low_top_p)
    assert isinstance(result, str)
    assert len(result) > 0


# Test run method with an empty task
def test_vllm_run_empty_task(vllm_instance):
    empty_task = ""
    result = vllm_instance.run(empty_task)
    assert isinstance(result, str)
    assert len(result) == 0


# Test initialization with invalid parameters
def test_vllm_invalid_init():
    with pytest.raises(ValueError):
        vLLM(
            model_name=None,
            tensor_parallel_size=-1,
            trust_remote_code="invalid",
            revision=123,
            temperature=-0.1,
            top_p=1.1,
        )


# Test running vLLM with a large number of parallel heads
def test_vllm_large_parallel_heads():
    vllm_instance = vLLM(tensor_parallel_size=16)
    task = "Large Parallel Heads Test"
    result = vllm_instance.run(task)
    assert isinstance(result, str)
    assert len(result) > 0


# Test running vLLM with trust_remote_code set to True
def test_vllm_trust_remote_code():
    vllm_instance = vLLM(trust_remote_code=True)
    task = "Trust Remote Code Test"
    result = vllm_instance.run(task)
    assert isinstance(result, str)
    assert len(result) > 0
