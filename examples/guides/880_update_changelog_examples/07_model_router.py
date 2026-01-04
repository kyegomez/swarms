"""
ModelRouter Example

This example demonstrates ModelRouter for intelligent model selection
and execution based on task requirements.
"""

from swarms import ModelRouter

router = ModelRouter(
    max_tokens=4000,
    temperature=0.5,
    max_workers=10,
)

task1 = "Analyze the sentiment and key themes in this customer feedback: The product is great but delivery was slow."
result1 = router.run(task1)

print("ModelRouter Result 1 (Sentiment Analysis):")
print(result1)

task2 = "Write a creative short story about a robot learning to paint"
result2 = router.run(task2)

print("\nModelRouter Result 2 (Creative Writing):")
print(result2)
