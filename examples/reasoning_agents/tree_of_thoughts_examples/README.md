# Tree of Thoughts Examples

Examples of **`TreeOfThoughts`** solving problems in mathematics, physics and general reasoning. Each problem has a checkable answer (given in the file's docstring), and each example uses a different combination of settings, so together they cover the whole API.

`TreeOfThoughts` implements [Tree of Thoughts (Yao et al., 2023)](https://arxiv.org/abs/2305.10601). Rather than answering in one pass, it grows a tree of partial solutions: it proposes candidate next steps, scores each one from 0 to 1, prunes weak branches, searches the tree, and writes the final answer from the best path. Every model output is a function call validated against a schema.

## Examples

### Mathematics

| Example | Problem | Search | Expected answer |
|---|---|---|---|
| [number_theory_counting.py](mathematics/number_theory_counting.py) | Count n ≤ 1000 with 7 dividing n² + n + 1 | BFS, beam of 3, 2 ratings averaged per candidate | 286 |
| [irrationality_proof.py](mathematics/irrationality_proof.py) | Prove √2 + √3 is irrational | DFS, strict `value_threshold=0.7`, custom `system_prompt` | A valid proof |
| [calculus_optimization.py](mathematics/calculus_optimization.py) | The 1 L can with the least surface area | BFS, `generation_strategy="sample"` | r ≈ 5.42 cm, h ≈ 10.84 cm, A ≈ 553.6 cm² |

### Physics

| Example | Problem | Search | Expected answer |
|---|---|---|---|
| [projectile_motion.py](physics/projectile_motion.py) | Where a ball thrown off a 20 m cliff lands | BFS, evaluator checks signs and units | ≈ 38.0 m |
| [loop_the_loop.py](physics/loop_the_loop.py) | Minimum release height for a loop of radius 0.8 m | DFS | 2.0 m (5R/2) |
| [fermi_estimation.py](physics/fermi_estimation.py) | Kinetic energy of all US cars at rush hour | BFS, `sample` + `evaluation_strategy="vote"`, 3 votes | ≈ 10¹³ J |

### Reasoning

| Example | Problem | Search | Expected answer |
|---|---|---|---|
| [knights_and_knaves.py](reasoning/knights_and_knaves.py) | Who is a knight and who is a knave | DFS case analysis | A knight, B knave, C knight |
| [constraint_scheduling.py](reasoning/constraint_scheduling.py) | Fit five talks into five slots under six rules | BFS, beam of 3 | Dana 9, Ben 10, Chen 11, Alice 13, Eli 14 |
| [bridge_and_torch.py](reasoning/bridge_and_torch.py) | Cross a bridge by torchlight in minimum time | DFS, `max_expansions=15` cost cap | 17 minutes |

## Running

```bash
export OPENAI_API_KEY="sk-..."
python examples/reasoning_agents/tree_of_thoughts_examples/physics/loop_the_loop.py
```

The examples use `gpt-5.4`, but any LiteLLM model that supports function calling works, e.g. `model_name="claude-sonnet-5"`. `run(task)` returns the answer; `agent.last_result` holds the best path (`steps`), the whole tree (`root`), whether a verified solution was found (`solved`), and the cost (`llm_calls`, `usage`).

## Choosing settings

| Setting | Use | When |
|---|---|---|
| `search_algorithm` | `"bfs"` | Several partial solutions are worth keeping at once (scheduling, multi-step calculations). |
| | `"dfs"` | Case analysis and planning, where you commit to a line and backtrack on a contradiction. |
| `generation_strategy` | `"propose"` | Constrained steps, where one call can list distinct options. |
| | `"sample"` | Open-ended steps, where independent calls give more variety. |
| `evaluation_strategy` | `"value"` | Steps can be checked on their own (arithmetic, logic, units). |
| | `"vote"` | Quality is relative, so comparing candidates beats rating them (estimation, writing). |

`thought_description` tells the model what one step looks like, and `evaluation_criteria` tells it how to judge progress. They adapt the search to a domain more than any other setting. Cost grows with `num_thoughts`, `breadth`, `max_depth` and `n_evaluate_samples`; use `max_expansions` to cap it.
