# Contributing to Swarms

<div align="left">
  <a href="https://swarms.world">
    <img src="https://github.com/kyegomez/swarms/blob/master/images/new_logo.png" style="margin: 15px; max-width: 500px" width="50%" alt="Swarms Logo">
  </a>
</div>

<p align="left">
  <em>The Enterprise-Grade Production-Ready Multi-Agent Orchestration Framework</em>
</p>

Swarms makes it simple to orchestrate agents to automate real-world work. Contributions of every size are welcome — the fastest way in is a `good first issue`.

| Where to help | What it looks like |
|---|---|
| Tests | Cover existing code in `swarms/`, add edge cases and integration tests |
| Docs | Fix docstrings, add examples to `examples/`, expand `docs/` |
| Swarm architectures | New multi-agent orchestration methods in `swarms/structs/` |
| Agents | New or improved specialized agents (finance, medical, code, research) |
| Cleanup | Delete dead code, remove duplicate implementations, simplify functions |
| Performance | Faster, cheaper swarm execution |

- [Good First Issues](https://github.com/kyegomez/swarms/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) · [Contributing Board](https://github.com/users/kyegomez/projects/1)

---

## Setup

```bash
git clone https://github.com/kyegomez/swarms.git
cd swarms
pip install -e .          # or: uv pip install -e .
```

Create a `.env` in the project root:

```bash
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
GROQ_API_KEY=""
WORKSPACE_DIR="agent_workspace"
```

[Environment setup docs →](https://docs.swarms.world/environment-setup)

**Layout**: `swarms/agents/` (agents) · `swarms/structs/` (swarms + workflows) · `swarms/tools/` · `swarms/prompts/` · `swarms/utils/` · `examples/` · `tests/` (mirrors `swarms/`) · `docs/`

---

## Reporting Issues

Search existing issues first. If it's new, open a [Bug Report](https://github.com/kyegomez/swarms/issues/new?template=bug_report.md) or [Feature Request](https://github.com/kyegomez/swarms/issues/new?template=feature_request.md) with a concise title, steps to reproduce, expected vs. actual behavior, and logs. Label it appropriately.

---

## Pull Requests

```bash
git checkout -b fix/short-description
# make the change, add a test
pytest tests/
git commit -am "Fix X in Y"
git push origin fix/short-description
```

Then open the PR against `master`, describe the problem it solves, and link the issue (`Fixes #1234`).

### Keep PRs short and simple

Review capacity is the bottleneck. PR size is the biggest factor in how fast your work merges — a small, obvious PR merges in hours; a large one can sit for weeks.

**Scope**

- One PR, one change: one bug fix, one feature, or one refactor. Find a second problem? Open a second PR.
- Fewer files is better. **Multi-file PRs take significantly longer to review** — the reviewer has to hold the whole change at once.
- Split large work into a sequence of small PRs that each stand alone and each leave the codebase working.
- No drive-by changes: don't mix reformatting, renames, dependency bumps, or unrelated cleanups into a functional PR.
- Don't reformat files you're editing. Change only the lines you need to; whitespace churn hides the real fix.

**Size**

| Diff | Expectation |
|---|---|
| < ~100 lines | Ideal — reviewed quickly |
| 100–300 lines | Fine if it's one coherent change |
| > ~300 lines | Slow review, or a request to split |
| Many unrelated files | Likely asked to split before review |

**Code**

- Smallest change that fixes the problem. Don't rewrite code you happen to be near.
- Reuse what exists instead of adding a parallel implementation.
- No speculative abstraction — no options or hooks for cases nobody asked for.
- No new dependencies unless unavoidable; justify them in the description.
- Delete dead code rather than deprecating it, in its own PR.

**Before opening**

- [ ] Diff contains only changes related to the stated purpose
- [ ] No large comment blocks or commented-out code
- [ ] Test added; `pytest tests/` passes
- [ ] `black` / `flake8` clean, no unrelated reformatting
- [ ] Description explains the problem and links the issue

A small, focused PR merges faster than a large one — even when the large one is better work.

---

## Coding Standards

- **Type annotations** on every function and method.
- **Docstrings** on every public class, function, and method ([Google style](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) or [NumPy](https://numpydoc.readthedocs.io/en/latest/format.html)): description, `Args`, `Returns`, `Raises`.
- **Tests** for every feature and bug fix, in `tests/` mirroring `swarms/`. Run with `pytest tests/`.
- **Style**: PEP 8, enforced with `black` and `flake8`. Match the surrounding code.
- **Docs**: update `docs/` when you change the public API.

### Comments

Write code that explains itself; comment only what it can't.

- **No multi-line comment blocks** — no banners, section dividers, ASCII art, or multi-paragraph explanations. Explanation belongs in the docstring.
- One short line for a non-obvious decision is the right size.
- No commented-out code — git history keeps it.
- Don't restate the code (`# loop over agents` above a `for` loop).

```python
# Bad — a block that restates the code and goes stale
# ------------------------------------------------------------
# This function takes a list of agents and runs each of them
# against the provided task, collecting the results into a
# list which is then returned to the caller.
# ------------------------------------------------------------
def run_all(agents: List[Agent], task: str) -> List[str]:
    return [agent.run(task) for agent in agents]

# Good — docstring for the contract, a comment only for the surprise
def run_all(agents: List[Agent], task: str) -> List[str]:
    """Run ``task`` on each agent and return their outputs in order."""
    # Sequential on purpose: agents share a rate-limited client.
    return [agent.run(task) for agent in agents]
```

---

## Resources

| | |
|---|---|
| Docs | [docs.swarms.world](https://docs.swarms.world) · [Quickstart](https://docs.swarms.world/quickstart) · [Agent API](https://docs.swarms.world/api/agent) |
| Examples | [examples/](https://github.com/kyegomez/swarms/tree/master/examples) — [single_agent](https://github.com/kyegomez/swarms/tree/master/examples/single_agent), [multi_agent](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent), [tools](https://github.com/kyegomez/swarms/tree/master/examples/tools) |
| Architectures | [SequentialWorkflow](https://docs.swarms.world/api/sequential-workflow) · [AgentRearrange](https://docs.swarms.world/api/agent-rearrange) · [MixtureOfAgents](https://docs.swarms.world/api/mixture-of-agents) · [GraphWorkflow](https://docs.swarms.world/api/graph-workflow) · [GroupChat](https://docs.swarms.world/api/group-chat) · [SwarmRouter](https://docs.swarms.world/api/swarm-router) |
| Community | [Discord](https://discord.gg/EamjgSaEQf) · [Twitter](https://twitter.com/kyegomez) · [LinkedIn](https://www.linkedin.com/company/the-swarm-corporation) · [YouTube](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) · [Events](https://lu.ma/swarms_calendar) · [Blog](https://medium.com/@kyeg) |
| Onboarding | [Book a session with the maintainer](https://cal.com/swarms/swarms-onboarding-session) |

Be respectful, give and take feedback openly, and collaborate. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

Contributions are licensed under the [Apache License](LICENSE). If you use Swarms in research, cite it via [CITATION.cff](./CITATION.cff).

**Happy contributing! 🚀**
