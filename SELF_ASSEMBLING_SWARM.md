# Self-Assembling Swarm: Coalition Architecture Search

A design for a swarm in which agents get a task and nothing else. They split the work, form teams, choose how each team works, connect teams to each other, and keep reorganising until they find the structure that solves the task best. Two things come out:

1. **An answer** to the task.
2. **A harness**: the best structure the swarm found (teams, roles, protocols and channels), saved as a static spec. It can be run again on similar tasks without paying to discover it again.

The algorithm is called **Coalition Architecture Search (CAS)**. Its core idea is that a multi-agent architecture is a *genome*: subgoals, teams, protocols and channels. The agents search over that genome themselves. Each agent makes local, self-interested moves (join, leave, recruit, merge, split, rewire), and the moves are scored against one shared, independently verified result.

Status: implemented in `swarms/structs/self_assembling_swarm.py` as `SelfAssemblingSwarm`.

---

## 1. Why a new algorithm

Fixed structures (sequential, concurrent, hierarchical, debate, vote) put the architecture choice on the user. LLM-designed swarms put it on one planner call that never sees the result of its choice. Bid-to-speak and leaderless message-board swarms let agents coordinate, but they don't form lasting groups or keep the structure that worked.

CAS differs on four points:

| Property | CAS |
|---|---|
| Who chooses the structure | The agents, through local moves |
| What drives the choice | Measured outcomes: verifier score, cost and per-agent credit |
| Unit of organisation | Teams (coalitions), each with its own internal protocol, linked by typed channels |
| What persists | Learned affinities, an archive of elite structures, and a compiled harness |

---

## 2. Concepts

| Term | Meaning |
|---|---|
| **Agent** `a` | An LLM agent with a persona, tools, a budget and a learned skill profile. |
| **Agora** | A shared blackboard holding the task, subgoals, team charters, beacons and the archive. |
| **Subgoal** `g` | A unit of work with a *kind* (e.g. `research`, `code`, `review`, `integrate`), a dependency list, and an output contract. |
| **Charter** `c` | A proposal to form a team: the subgoal, the roles needed, a protocol, and a size cap. |
| **Team** `t` | A coalition of agents that owns one subgoal and runs one protocol. |
| **Protocol** `p` | How a team works inside: drawn from a small grammar (Section 4). |
| **Port** | A team's typed interface: *needs* (inputs it waits for) and *offers* (outputs it produces). |
| **Channel** `e` | A link between two teams' ports, carrying inter-team messages. |
| **Liaison** | An agent that is a member of two teams, which bridges them. |
| **Genome** `Γ` | The whole architecture: `(G, T, P, E)`, i.e. subgoals, teams, protocols and channels. |
| **Verifier** `V` | An independent scorer of the final output. Tests, a rubric judge panel, or a reference check. Agents never score their own result. |
| **Epoch** | One full cycle of assemble, act, assess and adapt. |

### Learned state (persists across epochs)

| Symbol | Shape | Meaning |
|---|---|---|
| `W_ag[a, k]` | agents × subgoal kinds | How well agent `a` performs on kind `k` (EMA of credit). |
| `W_aa[a, b]` | agents × agents, in [-1, 1] | Trust: how well `a` and `b` do *together* (EMA of joint credit minus baseline). |
| `W_pk[p, k]` | protocols × kinds | How well protocol `p` works for subgoal kind `k`. |
| `W_ch[k1, k2]` | kinds × kinds | How much a channel between kinds `k1` and `k2` helps. |
| `τ` | scalar | Exploration temperature, annealed from `τ0` toward `τ_min`. |
| `Archive` | grid of genomes | MAP-Elites archive of the best genome per structural niche (Section 7). |

### Fitness

```
F(Γ) = Q − λ·Cost − μ·Latency
```

- `Q ∈ [0, 1]` is the verifier score. When the verifier is noisy, it is the mean of `r` repeated judgments.
- `Cost` is tokens spent, normalised to the budget.
- `Latency` is wall-clock time, normalised.
- `λ` and `μ` set how much the user cares about cost and speed against quality.

---

## 3. The algorithm

### 3.1 Overview

```
            ┌──────────────────────────────────────────────────────────┐
            │                         AGORA                            │
            │  task · subgoals · charters · beacons · archive · W_*    │
            └──────────────────────────────────────────────────────────┘
                 ▲            ▲             ▲              ▲
   ┌─────────────┴──┐ ┌───────┴──────┐ ┌────┴───────┐ ┌────┴────────┐
   │ 1. DECOMPOSE   │→│ 2. PITCH &   │→│ 3. WIRE    │→│ 4. ACT      │
   │ subgoal DAG    │ │    MATCH     │ │ ports →    │ │ protocols + │
   │ (merged drafts)│ │ teams form   │ │ channels   │ │ inter-team  │
   └────────────────┘ └──────────────┘ └────────────┘ │ messaging   │
          ▲                                            └─────┬───────┘
          │            ┌──────────────┐  ┌──────────────┐    │
          └────────────│ 6. ADAPT     │←─│ 5. ASSESS    │←───┘
                       │ moves, merge,│  │ verifier,    │
                       │ split, mutate│  │ credit       │
                       └──────────────┘  └──────────────┘
```

Epoch 1 runs all six phases. Later epochs skip phases 1 to 3 except where Phase 6 changed something: they *repair* the current genome instead of rebuilding it. Stability is what makes credit learnable.

### 3.2 Phase 1: Decompose

1. Each agent writes a short **draft decomposition** of the task in one cheap call: a list of subgoals with kinds, dependencies and output contracts.
2. The drafts are merged into one **subgoal DAG**. Near-duplicate subgoals are clustered by embedding similarity (cosine ≥ `σ_dup`), and one canonical subgoal is kept per cluster. **Support** is the number of drafts in the cluster.
3. Subgoals with support below `s_min` are dropped unless `τ` is high. Under high `τ`, minority ideas survive, which is a source of exploration.
4. An `integrate` subgoal is always added as the DAG's sink. Its needs are every leaf output, and its offer is the final answer.

Decomposition is itself part of the genome. Phase 6 can split a subgoal that keeps failing, or merge two that keep messaging each other.

### 3.3 Phase 2: Pitch and match (team formation)

**Self-assessment.** Each agent makes one forced tool call that returns a fit vector over all subgoals: `s_a(g) ∈ [0, 1]`, plus a one-line reason for each.

**Fit.** Self-assessment is blended with evidence, and the weight moves toward evidence as it accumulates:

```
fit(a, g) = α_a · s_a(g) + (1 − α_a) · W_ag[a, kind(g)],     α_a = 1 / (1 + n_a / κ)
```

Here `n_a` is the number of epochs of credit for `a` on that kind. Early on, agents are trusted about themselves; later, the record decides.

**Pitch.** For each subgoal, the top-`f` agents by fit may found a team. A founder writes a **charter**:

```json
{
  "charter_id": "c-07",
  "subgoal": "g-03",
  "founder": "a-12",
  "roles": [{"role": "implementer", "count": 2}, {"role": "critic", "count": 1}],
  "protocol": {"type": "CRITIC_LOOP", "rounds": 2},
  "size_cap": 4,
  "pitch": "Two implementers draft, one critic reviews against the contract."
}
```

The protocol the founder proposes is sampled from `softmax(W_pk[·, kind(g)] / τ)`, and the founder may override the sample with a justification. More than one charter may compete for the same subgoal; the losers dissolve in matching.

**Utility of joining.** For agent `a` and charter `c` (with current members `M_c`):

```
U(a, c) = fit(a, g_c)
        + β · mean_{b ∈ M_c} W_aa[a, b]                  # works well with these agents
        + γ · τ / sqrt(1 + familiarity(a, c))            # novelty bonus, fades as τ falls
        − δ · load(a)                                    # already in other teams
```

**Matching.** This is many-to-one deferred acceptance, a Gale–Shapley variant with capacities:

- Agents propose to charters in order of `U(a, c)`.
- Each charter keeps its best applicants, up to `size_cap`, ranked by role coverage and then by `fit(a, g_c) + β · mean W_aa` with the current holds. It rejects the rest.
- This repeats until no agent has a charter left to propose to.

The result is **stable**: no agent and charter both prefer each other over what they got. Charters with fewer than `min_size` members, or with an uncovered required role, dissolve. Their agents re-enter for one more round.

**Free agents.** Agents left unmatched become **scouts**. A scout can join any team mid-epoch on a `RECRUIT` request, or act as a liaison. An agent may belong to at most `m` teams (default 2). Being in two teams makes it a liaison between them.

A charter that wins its subgoal becomes team `t`. Competing charters for one subgoal are not merged: the one with the higher total matched utility wins.

### 3.4 Phase 3: Wire (inter-team topology)

Every team publishes a **port**, which comes from its subgoal's contract:

```json
{
  "team": "t-03",
  "needs":  [{"name": "api_schema", "type": "openapi"}],
  "offers": [{"name": "handlers", "type": "python_module"}]
}
```

Channels are created in three ways:

1. **Dependency channels.** A channel is created whenever one team's offer type matches another's need. These carry the DAG.
2. **Learned channels.** A channel between teams of kinds `k1` and `k2` is added with probability `sigmoid(W_ch[k1, k2])`. The swarm learns, for example, that `review` teams talking to `research` teams helps, even with no data dependency.
3. **Weak ties.** For exploration, a few random channels are added, with the count proportional to `τ`.

Liaisons create implicit channels that don't count against the message budget, but they do add to the liaison's load.

### 3.5 Phase 4: Act

All teams run concurrently. Each one runs its protocol (Section 4) on its subgoal. A team waiting on a need either:

- **blocks** until the need arrives (default for hard dependencies); or
- **proceeds on an assumption**, writes the assumption into its output, and reconciles when the real input arrives. This is allowed when the wait exceeds `t_wait`, and it prevents deadlock.

**Inter-team messages.** Each message has a type:

| Type | Purpose |
|---|---|
| `REQUEST` | Ask a team for something on its port, with a deadline. |
| `OFFER` | Announce an artifact matching another team's need. |
| `ARTIFACT` | Deliver the artifact (stored once in the agora; messages carry a reference). |
| `CHALLENGE` | Cross-team critique: a claim about another team's output, with evidence. The target team must answer, either with a fix or with a rebuttal. |
| `HANDOFF` | Pass a piece of work to a team better placed to do it. |
| `RECRUIT` | Ask for a scout, or for a member of another team, with a role. |
| `BEACON` | Broadcast to the agora: progress, blockers and confidence. It's cheap and has no reply. |

```json
{
  "id": "m-221", "type": "CHALLENGE", "from": "t-05", "to": "t-03",
  "epoch": 3, "ref": "artifact:t-03/handlers@2",
  "body": "Handler for /orders ignores the 'limit' field the schema requires.",
  "evidence": "schema.paths['/orders'].get.parameters[1]"
}
```

**Message budget.** Each team gets `B_msg` messages per epoch, and the tokens they cost are charged to `F`. Because chatter costs fitness, the swarm learns how much talking pays. A team that runs out can still read, and can still send `BEACON`.

**Routing.** Point-to-point messages travel only on existing channels. A team that needs a partner it has no channel to sends a `REQUEST` to the agora. If a team answers it, the pair gets a provisional channel. If that channel is used more than `u_min` times, it becomes a learned channel for the next epoch.

**Integration.** The `integrate` team is formed like any other team, through charters and matching. Its protocol decides how the leaf outputs are combined.

### 3.6 Phase 5: Assess (verification and credit)

1. **Score.** `V` scores the final output and gives `Q`, and the swarm computes `F(Γ)`. For noisy judges, scoring runs `r` times and takes the mean with a confidence interval.
2. **Team credit** comes from leave-one-team-out ablation over *cached* outputs. Only the integrator and the verifier re-run:
   ```
   Δ_t = F(Γ) − F(Γ with t's output replaced by ∅ or by t's previous-best output)
   ```
   This costs `|T|` integrator-plus-verifier calls, not a full re-run.
3. **Agent credit within a team** depends on the protocol:
   - `CHAIN` and `CRITIC_LOOP`: the verifier gives a cheap partial score to each intermediate artifact, and each agent gets the change in score its step caused.
   - `FANOUT`, `VOTE` and `DEBATE`: leave one member's output out at the merge or judge step.
   - Every protocol also gets **calibrated peer review**: members rate each other, and each rater's weight is its historic agreement with the verifier. Raters who flatter lose weight.
   - The team's `Δ_t` is split by these shares: `credit(a) = Δ_t · share_t(a)`.
4. **Channel credit.** Drop a channel's messages and re-run the receiving team's last step from cache. Do this only for channels that carried a `CHALLENGE` or `HANDOFF` (the costly ones), and sample `n_ch` of them per epoch.

### 3.7 Phase 6: Adapt (learning and restructuring)

**Update learned state** with an EMA at rate `η`:

```
W_ag[a, kind(g)] ← (1−η)·W_ag + η·credit(a)
W_aa[a, b]       ← (1−η)·W_aa + η·(Δ_t − baseline_kind)      for a, b in the same team t
W_pk[p, kind(g)] ← (1−η)·W_pk + η·(Δ_t − baseline_kind)
W_ch[k1, k2]     ← (1−η)·W_ch + η·channel_credit
```

`baseline_kind` is the running mean `Δ` for that subgoal kind, so the updates measure *better than usual*, not just *good*.

**Agent moves.** Each agent chooses one move with probability `softmax(ΔÛ / τ)`, where `ΔÛ` is its estimated gain in utility from Section 3.3:

| Move | Effect |
|---|---|
| `STAY` | No change. Its `ΔÛ` is 0, which is the reference. |
| `LEAVE` | Become a scout. |
| `JOIN c` | Apply to a team with an open role. The team accepts by its ranking rule. |
| `SWAP b` | Trade places with an agent in another team, if both agree. |
| `FOUND g` | Charter a new team for a subgoal that is unowned or underperforming. |

At most a fraction `ρ` of agents may move per epoch, picked by highest `|ΔÛ|`. This keeps the structure stable enough for credit to mean something.

**Team moves** are triggered by measured signals, not by asking an LLM:

| Move | Trigger |
|---|---|
| `MERGE t1, t2` | Traffic between the two teams divided by traffic inside them exceeds `θ_merge`, and the combined size fits within the cap. Two teams that talk all the time are one team. |
| `SPLIT t` | The team has more than `2·min_size` members, and trust inside it splits into two clusters (spectral cut of `W_aa` restricted to the team), or the team's credit share has high variance with a near-zero tail. |
| `SPLIT g` | A subgoal's `Q` share has stayed below `q_low` for `k_fail` epochs. Its team re-decomposes it into two subgoals. |
| `MUTATE p` | Change the team's protocol. This happens with probability `p_mut · τ`, sampled from `W_pk`. |
| `DISSOLVE t` | `Δ_t ≤ 0` for `k_fail` epochs: the team adds nothing. Its subgoal goes back to the pool, or is dropped if no other team's need refers to it. |
| `REWIRE` | Prune channels with negative credit, and promote provisional channels that carried more than `u_min` messages. |

**Archive and restart.** Insert `Γ` into the archive (Section 7). With probability `p_restart`, the next epoch starts from an archive elite in a *different* niche instead of from the current genome, which escapes local optima.

**Anneal.** `τ ← max(τ_min, τ · decay)`.

### 3.8 Termination and output

The search stops at the first of:

- **Plateau:** the best `F` improved by less than `ε` over `k_plateau` epochs.
- **Stable structure:** the genome edit distance between consecutive epochs is below `δ_struct` for `k_plateau` epochs.
- **Budget:** the token or time budget is spent.
- **Verifier pass:** for verifiers with a pass/fail result, such as tests, `Q = 1` while `Cost` is below the target.

The swarm then returns:

1. **The answer:** the final output of the best genome seen.
2. **The harness:** the best genome, compiled (Section 6).
3. **The Pareto front** of `Q` against cost from the archive, so the user can choose a cheaper harness.
4. **A report:** learned affinities, per-team credit, and which moves helped.

---

## 4. Protocol grammar

Teams choose from a small, composable set. Keeping the set small keeps `W_pk` learnable.

| Protocol | Parameters | Inside the team |
|---|---|---|
| `SOLO` | – | One agent does the subgoal. |
| `CHAIN` | order | Each member refines the previous member's output. |
| `FANOUT` | merger | All members work independently, and one member merges the results. |
| `CRITIC_LOOP` | maker(s), critic, rounds | The maker drafts, the critic reviews against the contract, and the loop repeats until the critic passes the draft or the rounds run out. |
| `DEBATE` | sides, judge, rounds | Members argue positions, and a judge member rules. |
| `VOTE` | quorum | Members answer independently, and the answer with a majority or plurality wins. |
| `LEAD` | lead | One member splits the subgoal among the others and assembles their parts. |

Protocols nest: a `LEAD` team's sub-pieces may each run `CRITIC_LOOP`. Nesting deeper than one level counts as a `SPLIT g` and creates real child teams, so it stays visible in the genome.

---

## 5. Pseudocode

```python
def coalition_architecture_search(task, agents, verifier, budget, cfg):
    agora = Agora(task)
    state = LearnedState(agents, cfg)           # W_ag, W_aa, W_pk, W_ch, tau
    archive = EliteArchive(cfg.niches)
    genome = None

    while not done(archive, budget, cfg):
        if genome is None or genome.needs_rebuild:
            subgoals = decompose(task, agents, state.tau, cfg)          # 3.2
            charters = pitch(subgoals, agents, state, cfg)              # 3.3
            teams = deferred_acceptance(agents, charters, state, cfg)
            channels = wire(teams, state, cfg)                          # 3.4
            genome = Genome(subgoals, teams, channels)
        else:
            genome = genome.repair()                                    # apply last epoch's moves

        trace = act(genome, agora, cfg.msg_budget, cfg.t_wait)          # 3.5
        score = verifier.score(trace.final_output, repeats=cfg.r)
        fitness = score - cfg.lam * trace.cost - cfg.mu * trace.latency

        credit = assign_credit(genome, trace, verifier, cfg)            # 3.6
        state.update(genome, credit)                                    # 3.7
        archive.insert(genome, fitness, trace.final_output)

        moves = choose_agent_moves(agents, genome, state, cfg.rho)
        moves += detect_team_moves(genome, trace, state, cfg)
        genome = genome.with_moves(moves)

        if random() < cfg.p_restart:
            genome = archive.sample_elite(other_niche_than=genome)

        state.tau = max(cfg.tau_min, state.tau * cfg.decay)

    best = archive.best()
    return best.output, compile_harness(best.genome), archive.pareto(), state.report()
```

---

## 6. Compiling the harness

The best genome is frozen into a static, serialisable spec. A frozen harness runs with no decompose, pitch, match or assess overhead.

```json
{
  "harness": "cas-2026-11-20-a",
  "fitness": {"Q": 0.91, "cost": 0.34},
  "teams": [
    {"id": "research", "subgoal_kind": "research", "protocol": {"type": "FANOUT", "merger": "R1"},
     "members": [{"name": "R1", "persona": "..."}, {"name": "R2", "persona": "..."}]},
    {"id": "build", "subgoal_kind": "code", "protocol": {"type": "CRITIC_LOOP", "rounds": 2},
     "members": [{"name": "B1", "role": "maker"}, {"name": "B2", "role": "critic"}]},
    {"id": "integrate", "subgoal_kind": "integrate", "protocol": {"type": "SOLO"},
     "members": [{"name": "I1"}]}
  ],
  "channels": [
    {"from": "research", "to": "build", "kind": "dependency"},
    {"from": "build", "to": "research", "kind": "challenge", "max_messages": 3}
  ],
  "liaisons": [{"agent": "R2", "teams": ["research", "build"]}]
}
```

**Compile targets.**

- **DAG part:** each team becomes a node or subgraph built from its protocol, and dependency channels become edges. This maps onto a graph workflow.
- **Feedback channels** (`CHALLENGE` that flows backward): these become a bounded loop. Either the receiving team runs a `CRITIC_LOOP` with the sender as an extra critic, or the loop is unrolled to the channel's `max_messages`.
- **Personas:** saved as the agents' system prompts, plus the skill notes they learned while working (`W_ag` reasons and peer-review comments).

**Re-use on new tasks.** A harness is keyed by the task's *kind signature*: the multiset of subgoal kinds in its DAG. On a new task, the swarm first runs one cheap decomposition. If the result matches an archived signature, it warm-starts from that harness with a low `τ`, so it fine-tunes instead of searching from scratch.

---

## 7. Diversity: the elite archive

Without pressure toward diversity, the search collapses onto the first structure that works. The archive is a MAP-Elites grid over structural descriptors:

| Descriptor | Buckets |
|---|---|
| Number of teams | 1, 2–3, 4–6, 7+ |
| Depth of the team DAG | 1, 2, 3, 4+ |
| Inter-team messages per epoch | low, mid, high |
| Cost | quartiles of budget |

Each cell keeps the best genome that falls into it. Restarts (`p_restart`) sample from cells other than the current one. The final Pareto front is read from the whole grid.

---

## 8. Two operating modes

| Mode | When | How fitness is measured |
|---|---|---|
| **Single task** | One task, no task family | Every epoch is a real attempt, and the answer improves epoch by epoch. Credit comes only from that task, so the harness is tuned to it. |
| **Task family** | Many similar tasks, such as tickets, filings or repos | Each epoch runs on a minibatch of `b` *probe* tasks, and `F` is the mean. After convergence, the frozen harness runs on the rest. This mode gives a harness that generalises, and it is the main use. |

In task-family mode, credit is averaged over the minibatch, which cuts noise a lot. Rotate the probe tasks so the structure doesn't overfit to a few of them.

---

## 9. Cost

Per epoch, with `N` agents, `|G|` subgoals, `|T|` teams and act cost `C_act`:

| Step | LLM calls |
|---|---|
| Decompose (epoch 1, or on rebuild) | `N` drafts + 1 merge |
| Self-assessment | `N`. Each call returns a vector over all subgoals. |
| Charters | `≤ f·|G|` |
| Matching | 0. It runs on numbers. |
| Act | `C_act`, which dominates |
| Verify | `r` |
| Team credit | `|T|·(1 + r)` integrator-plus-verifier re-runs |
| Agent credit | Partial scores on intermediate artifacts, plus one peer-review call per team |
| Adapt | 0, except for `SPLIT g`, which costs one re-decomposition |

A repair epoch costs about `C_act + (|T| + 1)·(1 + r)` plus the peer-review calls. Overhead is usually 20 to 40 percent on top of acting. Once the harness is frozen, that overhead is gone.

---

## 10. Defaults

| Knob | Default | Notes |
|---|---|---|
| `N` agents | 8–16 | Fewer than 6 leaves too little room to organise. |
| Epochs | 6–12 | Stops earlier on a plateau. |
| `min_size`, `size_cap` | 1, 5 | |
| `m` (teams per agent) | 2 | |
| `f` (founders per subgoal) | 2 | |
| `κ` (self-report decay) | 3 | |
| `β, γ, δ` | 0.5, 0.3, 0.2 | |
| `ρ` (movers per epoch) | 0.25 | |
| `τ0, τ_min, decay` | 1.0, 0.05, 0.7 | |
| `η` (EMA rate) | 0.3 | |
| `B_msg` | 12 per team | |
| `θ_merge` | 1.0 | |
| `p_mut, p_restart` | 0.3, 0.1 | |
| `r` (verifier repeats) | 1 for tests, 3 for LLM judges | |
| `λ, μ` | 0.2, 0.05 | |

---

## 11. Failure modes and safeguards

| Risk | Safeguard |
|---|---|
| **Thrash:** the structure changes every epoch, so credit means nothing. | The `ρ` cap on movers, a repair-not-rebuild default, and annealing `τ`. |
| **Herding:** every agent joins the one strong team. | `size_cap`, the `δ·load` term, and the archive's diversity pressure. |
| **Free-riding:** an agent sits in good teams and adds nothing. | Within-team credit, and `W_ag` that drifts down, so teams stop accepting it. |
| **Goodhart / collusion against the judge** | The verifier is outside the swarm. Prefer objective verifiers (tests, reference checks). Rotate judge panels, and hold out probe tasks in task-family mode. |
| **Flattering peer reviews** | Peer weight is calibrated against the verifier, so flatterers lose weight. |
| **Message storms** | `B_msg`, message tokens charged to `F`, and `BEACON` as the only broadcast. |
| **Deadlock on dependencies** | `t_wait`, after which a team may proceed on an assumption and reconcile later. |
| **Noisy verifier** | Repeats `r`, rank-based selection in the archive, and minibatches in task-family mode. |
| **Overfitting the harness to one task** | Task-family mode with rotating probes, and checking the harness on held-out tasks. |

---

## 12. Worked example

Task family: *"Implement the endpoint described in this ticket so the hidden tests pass."* There are 10 agents with mixed personas. The verifier is the test suite (`Q` is the pass rate) plus a lint check.

- **Epoch 1.** The drafts merge into five subgoals: `read-spec`, `schema`, `handlers`, `tests-review` and `integrate`. Matching forms five teams. `handlers` runs `FANOUT` with three members. `Q = 0.52`. Team credit shows `tests-review` with `Δ ≈ 0`.
- **Epoch 2.** `tests-review` has a negative `Δ` again. The `schema` and `handlers` teams exchanged 14 messages against 5 internal ones, which is over `θ_merge`. Two agents leave `tests-review`. One founds a `CHALLENGE`-heavy `edge-cases` team; the other becomes a liaison. `Q = 0.66`.
- **Epoch 3.** `MERGE schema + handlers` into one `build` team. A protocol mutation turns it from `FANOUT` into `CRITIC_LOOP`, and `W_pk[CRITIC_LOOP, code]` rises. `Q = 0.81`.
- **Epoch 4.** `DISSOLVE tests-review`. The learned channel `edge-cases → build` carries 3 challenges, each worth about +0.04 `Q`. `Q = 0.88`.
- **Epochs 5–6.** The structure is stable, and the gain is below `ε`. Stop.

**Compiled harness:**

- `read-spec` (`SOLO`) → `build` (`CRITIC_LOOP`, 2 rounds) ⇄ `edge-cases` (`FANOUT`, 2 agents, challenge channel capped at 3) → `integrate` (`SOLO`).
- It costs 38% less than epoch 1's structure and scores 0.88 on held-out tickets.

---

## 13. How to evaluate the algorithm

**Baselines,** all with the same token budget:

- A single agent.
- Each fixed structure: sequential, concurrent, hierarchical, debate, vote, mixture.
- A one-shot, LLM-designed architecture.
- Random architecture search over the same genome space and protocol grammar.
- CAS with learning turned off.

**Metrics:**

- `Q` at a fixed budget, and budget to reach a target `Q`.
- Wall-clock time.
- **Transfer:** the `Q` of the frozen harness on held-out tasks in the same family.
- Stability: the number of epochs to converge.

**Ablations:** remove one part at a time:

- trust (`W_aa`);
- inter-team channels (dependency channels only);
- `CHALLENGE` messages;
- merge and split;
- protocol mutation;
- the archive and restarts;
- annealing (`τ` fixed);
- calibrated peer review.

**Claim to test:** on task families with real structure (coding, research reports, due diligence), CAS's harness beats the best fixed structure chosen *after the fact*, at lower cost once frozen.

---

## 14. Proposed API sketch

```python
from swarms import Agent, SelfAssemblingSwarm

swarm = SelfAssemblingSwarm(
    agents=[Agent(agent_name=f"A{i}", model_name="gpt-5.4", max_loops=1) for i in range(12)],
    verifier=run_hidden_tests,            # callable(output) -> float in [0, 1]
    max_epochs=10,
    budget_tokens=2_000_000,
    cost_weight=0.2,
    mode="task_family",
)

result = swarm.run(tasks=probe_tickets)
result.answer          # best output, or one per probe task
result.harness         # compiled spec (Section 6)
result.pareto          # quality against cost from the archive
result.report          # affinities, credit, which moves paid off

harness = swarm.load_harness(result.harness)
harness.run(new_ticket)                 # frozen, no search overhead
```

---

## 15. Open questions

1. **Kind vocabulary.** Should subgoal kinds be a fixed list, or clusters learned from decompositions? Learned clusters generalise better, but they make `W_pk` sparse at first.
2. **Credit for liaisons.** Should a liaison be credited through both teams, or through the channel it carries?
3. **Verifier cost.** When `V` is an expensive LLM panel, should a cheap proxy judge do the partial scoring and the panel judge only final candidates?
4. **Heterogeneous models.** Agents on different models make `W_ag` partly a model-selection signal. Should the model be a separate gene, so the search can also pick models per role?
5. **Online harnesses.** Should a frozen harness keep a small `τ` in production, and keep learning from live verifier signals?
