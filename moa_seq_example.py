from swarms.structs.self_moa_seq import SelfMoASeq

# Initialize
moa_seq = SelfMoASeq(
    model_name="gpt-4o-mini",
    temperature=0.7,
    window_size=6,
    verbose=True,
    num_samples=4,
)

# Run
task = (
    "Describe an effective treatment plan for a patient with a broken rib. "
    "Include immediate care, pain management, expected recovery timeline, and potential complications to watch for."
)

result = moa_seq.run(task)
print(result)
