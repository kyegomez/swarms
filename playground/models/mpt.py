from swarms.models.mpt import MPT

mpt_instance = MPT(
    "mosaicml/mpt-7b-storywriter",
    "EleutherAI/gpt-neox-20b",
    max_tokens=150,
)

mpt_instance.generate("Once upon a time in a land far, far away...")
