# CLI LLM Council Guide: Collaborative Multi-Agent Decisions

Run the LLM Council directly from command line for collaborative decision-making with multiple AI agents through peer review and synthesis.

## Overview

The LLM Council creates a collaborative environment where:

1. **Multiple Perspectives**: Each council member (GPT-5.1, Gemini, Claude, Grok) independently responds
2. **Peer Review**: Members evaluate and rank each other's anonymized responses
3. **Synthesis**: A Chairman synthesizes the best elements into a final answer

---

## Basic Usage

### Step 1: Run a Simple Query

```bash
swarms llm-council --task "What are the best practices for code review?"
```

### Step 2: Enable Verbose Output

```bash
swarms llm-council --task "How should we approach microservices architecture?" --verbose
```

### Step 3: Process the Results

The council returns:
- Individual member responses
- Peer review rankings
- Synthesized final answer

---

## Use Case Examples

### Strategic Business Decisions

```bash
swarms llm-council --task "Should our SaaS startup prioritize product-led growth or sales-led growth? Consider market size, CAC, and scalability."
```

### Technology Evaluation

```bash
swarms llm-council --task "Compare Kubernetes vs Docker Swarm for a startup with 10 microservices. Consider cost, complexity, and scalability."
```

### Investment Analysis

```bash
swarms llm-council --task "Evaluate investment opportunities in AI infrastructure companies. Consider market size, competition, and growth potential."
```

### Policy Analysis

```bash
swarms llm-council --task "What are the implications of implementing AI regulation similar to the EU AI Act in the United States?"
```

### Research Questions

```bash
swarms llm-council --task "What are the most promising approaches to achieving AGI? Evaluate different research paradigms."
```

---

## Council Members

The default council includes:

| Member | Model | Strengths |
|--------|-------|-----------|
| **GPT-5.1 Councilor** | gpt-5.1 | Analytical, comprehensive |
| **Gemini 3 Pro Councilor** | gemini-3-pro | Concise, well-processed |
| **Claude Sonnet 4.5 Councilor** | claude-sonnet-4.5 | Thoughtful, balanced |
| **Grok-4 Councilor** | grok-4 | Creative, innovative |
| **Chairman** | gpt-5.1 | Synthesizes final answer |

---

## Workflow Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   GPT-5.1   │  Gemini 3   │ Claude 4.5  │   Grok-4    │
│  Councilor  │  Councilor  │  Councilor  │  Councilor  │
└─────────────┴─────────────┴─────────────┴─────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Anonymized Peer Review                         │
│         Each member ranks all responses (anonymized)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Chairman                                  │
│         Synthesizes best elements from all responses             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Final Synthesized Answer                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Best Practices

!!! tip "Query Formulation"
    - Be specific and detailed in your queries
    - Include context and constraints
    - Ask for specific types of analysis

!!! tip "When to Use LLM Council"
    - Complex decisions requiring multiple perspectives
    - Research questions needing comprehensive analysis
    - Strategic planning and evaluation
    - Questions with trade-offs to consider

!!! tip "Performance Tips"
    - Use `--verbose` for detailed progress tracking
    - Expect responses to take 30-60 seconds
    - Complex queries may take longer

!!! warning "Limitations"
    - Requires multiple API calls (higher cost)
    - Not suitable for simple factual queries
    - Response time is longer than single-agent queries

---

## Command Reference

```bash
swarms llm-council --task "<query>" [--verbose]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--task` | string | **Required** | Query for the council |
| `--verbose` | flag | False | Enable detailed output |

---

## Next Steps

- [CLI Heavy Swarm Guide](./cli_heavy_swarm_guide.md) - Complex task analysis
- [CLI Reference](./cli_reference.md) - Complete command documentation
- [LLM Council Python API](../examples/llm_council_quickstart.md) - Programmatic usage

