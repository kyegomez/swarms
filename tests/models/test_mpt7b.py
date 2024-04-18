import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from swarms.models.mpt import MPT7B


def test_mpt7b_init():
    mpt = MPT7B(
        "mosaicml/mpt-7b-storywriter",
        "EleutherAI/gpt-neox-20b",
        max_tokens=150,
    )

    assert isinstance(mpt, MPT7B)
    assert mpt.model_name == "mosaicml/mpt-7b-storywriter"
    assert mpt.tokenizer_name == "EleutherAI/gpt-neox-20b"
    assert isinstance(mpt.tokenizer, AutoTokenizer)
    assert isinstance(mpt.model, AutoModelForCausalLM)
    assert mpt.max_tokens == 150


def test_mpt7b_run():
    mpt = MPT7B(
        "mosaicml/mpt-7b-storywriter",
        "EleutherAI/gpt-neox-20b",
        max_tokens=150,
    )
    output = mpt.run(
        "generate", "Once upon a time in a land far, far away..."
    )

    assert isinstance(output, str)
    assert output.startswith("Once upon a time in a land far, far away...")


def test_mpt7b_run_invalid_task():
    mpt = MPT7B(
        "mosaicml/mpt-7b-storywriter",
        "EleutherAI/gpt-neox-20b",
        max_tokens=150,
    )

    with pytest.raises(ValueError):
        mpt.run(
            "invalid_task",
            "Once upon a time in a land far, far away...",
        )


def test_mpt7b_generate():
    mpt = MPT7B(
        "mosaicml/mpt-7b-storywriter",
        "EleutherAI/gpt-neox-20b",
        max_tokens=150,
    )
    output = mpt.generate("Once upon a time in a land far, far away...")

    assert isinstance(output, str)
    assert output.startswith("Once upon a time in a land far, far away...")


def test_mpt7b_batch_generate():
    mpt = MPT7B(
        "mosaicml/mpt-7b-storywriter",
        "EleutherAI/gpt-neox-20b",
        max_tokens=150,
    )
    prompts = ["In the deep jungles,", "At the heart of the city,"]
    outputs = mpt.batch_generate(prompts, temperature=0.7)

    assert isinstance(outputs, list)
    assert len(outputs) == len(prompts)
    for output in outputs:
        assert isinstance(output, str)


def test_mpt7b_unfreeze_model():
    mpt = MPT7B(
        "mosaicml/mpt-7b-storywriter",
        "EleutherAI/gpt-neox-20b",
        max_tokens=150,
    )
    mpt.unfreeze_model()

    for param in mpt.model.parameters():
        assert param.requires_grad
