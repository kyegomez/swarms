"""
SelfMoASeq Example

This example demonstrates SelfMoASeq (Self-Mixture of Agents Sequential)
for generating multiple outputs and aggregating them sequentially using
a sliding window approach.
"""

from swarms import SelfMoASeq

self_moa = SelfMoASeq(
    model_name="gpt-4o-mini",
    temperature=0.7,
    window_size=6,
    reserved_slots=3,
    num_samples=10,
    max_loops=5,
    verbose=True,
)

task = "Write a comprehensive analysis of the benefits and challenges of renewable energy"
result = self_moa.run(task=task)

print("SelfMoASeq Result:")
print(result)
