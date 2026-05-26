# LLM Council Quickstart

`LLMCouncil` lets several agents review the same problem and synthesize a stronger final answer. Use it when you want multiple perspectives, peer review, or a decision that benefits from explicit critique.

For CLI usage, see the [CLI LLM Council Guide](../cli/cli_llm_council_guide.md). For the full class reference, see [LLM Council](../structs/llm_council.md).

## Minimal Example

```python
from swarms import LLMCouncil

council = LLMCouncil(
    name="Product-Review-Council",
    model_name="gpt-4o-mini",
    max_loops=1,
)

result = council.run(
    "Evaluate whether we should launch a self-serve onboarding flow this quarter."
)

print(result)
```

## When to Use It

LLM Council works well for:

- Product or strategy decisions with tradeoffs.
- Technical reviews that need several perspectives.
- Risk analysis where one agent may miss an edge case.
- Draft review before content is sent to users.

For deeper examples, see [LLM Council Examples](llm_council_examples.md).

## Practical Tips

- Keep the task specific so the council can compare opinions clearly.
- Use a small number of council members for fast iteration.
- Ask for evidence, risks, and a final recommendation in the prompt.
- Review the final synthesis before using it in high-stakes workflows.

## Related Guides

- [LLM Council Reference](../structs/llm_council.md)
- [LLM Council Examples](llm_council_examples.md)
- [Council as Judge Example](council_as_judge_example.md)
- [Majority Voting Example](majority_voting_example.md)
- [CLI LLM Council Guide](../cli/cli_llm_council_guide.md)
