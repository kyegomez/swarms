import pytest

from swarms.models import (
    GPT4VisionAPI,
    HuggingfaceLLM,
    Mixtral,
    ZeroscopeTTV,
)
from swarms.structs.model_parallizer import ModelParallelizer

# Initialize the models
custom_config = {
    "quantize": True,
    "quantization_config": {"load_in_4bit": True},
    "verbose": True,
}
huggingface_llm = HuggingfaceLLM(
    model_id="NousResearch/Nous-Hermes-2-Vision-Alpha",
    **custom_config,
)
mixtral = Mixtral(load_in_4bit=True, use_flash_attention_2=True)
gpt4_vision_api = GPT4VisionAPI(max_tokens=1000)
zeroscope_ttv = ZeroscopeTTV()


def test_init():
    mp = ModelParallelizer(
        [
            huggingface_llm,
            mixtral,
            gpt4_vision_api,
            zeroscope_ttv,
        ]
    )
    assert isinstance(mp, ModelParallelizer)


def test_run():
    mp = ModelParallelizer([huggingface_llm])
    result = mp.run(
        "Create a list of known biggest risks of structural collapse"
        " with references"
    )
    assert isinstance(result, str)


def test_run_all():
    mp = ModelParallelizer(
        [
            huggingface_llm,
            mixtral,
            gpt4_vision_api,
            zeroscope_ttv,
        ]
    )
    result = mp.run_all(
        "Create a list of known biggest risks of structural collapse"
        " with references"
    )
    assert isinstance(result, list)
    assert len(result) == 5


def test_add_llm():
    mp = ModelParallelizer([huggingface_llm])
    mp.add_llm(mixtral)
    assert len(mp.llms) == 2


def test_remove_llm():
    mp = ModelParallelizer([huggingface_llm, mixtral])
    mp.remove_llm(mixtral)
    assert len(mp.llms) == 1


def test_save_responses_to_file(tmp_path):
    mp = ModelParallelizer([huggingface_llm])
    mp.run(
        "Create a list of known biggest risks of structural collapse"
        " with references"
    )
    file = tmp_path / "responses.txt"
    mp.save_responses_to_file(file)
    assert file.read_text() != ""


def test_get_task_history():
    mp = ModelParallelizer([huggingface_llm])
    mp.run(
        "Create a list of known biggest risks of structural collapse"
        " with references"
    )
    assert mp.get_task_history() == [
        "Create a list of known biggest risks of structural collapse"
        " with references"
    ]


def test_summary(capsys):
    mp = ModelParallelizer([huggingface_llm])
    mp.run(
        "Create a list of known biggest risks of structural collapse"
        " with references"
    )
    mp.summary()
    captured = capsys.readouterr()
    assert "Tasks History:" in captured.out


def test_enable_load_balancing():
    mp = ModelParallelizer([huggingface_llm])
    mp.enable_load_balancing()
    assert mp.load_balancing is True


def test_disable_load_balancing():
    mp = ModelParallelizer([huggingface_llm])
    mp.disable_load_balancing()
    assert mp.load_balancing is False


def test_concurrent_run():
    mp = ModelParallelizer([huggingface_llm, mixtral])
    result = mp.concurrent_run(
        "Create a list of known biggest risks of structural collapse"
        " with references"
    )
    assert isinstance(result, list)
    assert len(result) == 2


def test_concurrent_run_no_task():
    mp = ModelParallelizer([huggingface_llm])
    with pytest.raises(TypeError):
        mp.concurrent_run()


def test_concurrent_run_non_string_task():
    mp = ModelParallelizer([huggingface_llm])
    with pytest.raises(TypeError):
        mp.concurrent_run(123)


def test_concurrent_run_empty_task():
    mp = ModelParallelizer([huggingface_llm])
    result = mp.concurrent_run("")
    assert result == [""]
