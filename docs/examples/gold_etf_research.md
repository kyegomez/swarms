# Gold ETF Research Example

The Swarms examples include finance-oriented workflows that can research exchange-traded funds, compare market instruments, and produce structured investment notes.

Use this page as a starting point for adapting Swarms to financial research tasks. This is documentation for building agent workflows, not financial advice.

## Example Pattern

A typical ETF research swarm has:

- A research agent for gathering fund context
- A risk agent for volatility, liquidity, and concentration concerns
- A comparison agent for ranking candidate ETFs
- A final writer or judge agent that produces a concise report

## Prompt Shape

```text
Compare the top gold ETFs for a long-term portfolio.
Include ticker, expense ratio, liquidity considerations,
tracking approach, and key risks. Return a markdown table.
```

## Related Examples

- Heavy swarm examples: <https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/heavy_swarm_examples>
- RAG examples with gold ETF sample data: <https://github.com/kyegomez/swarms/tree/master/examples/single_agent/rag>
- LLM Council ETF examples: <https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/llm_council_examples>

## Related Docs

- [LLM Council Examples](../swarms/examples/llm_council_examples.md)
- [HeavySwarm](../swarms/structs/heavy_swarm.md)
- [Agent with Tools](../swarms/examples/agent_with_tools.md)
