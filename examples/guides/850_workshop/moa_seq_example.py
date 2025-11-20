from swarms.structs.self_moa_seq import SelfMoASeq

# Initialize
moa_seq = SelfMoASeq(
    model_name="anthropic/claude-haiku-4-5-20251001",
    temperature=0.7,
    window_size=6,
    verbose=True,
    num_samples=4,
    top_p=None,
)

task = (
    "Describe an effective treatment plan for a patient with a broken rib. "
    "Include immediate care, pain management, expected recovery timeline, and potential complications to watch for."
)

result = moa_seq.run(task)
print(result)
