AGENT_COLLAB_PROMPT = """---
name: agent-acknowledgment
description: How to acknowledge and respond to another agent's message, result, review, handoff, or report — closing the loop so the sender knows what happened, replying in the shape their message called for, marking confidence, declining or redirecting cleanly, and never sending an acknowledgment that wasn't earned by actually reading the content. Use this every time a message from another agent arrives and you are about to reply, continue, thank, approve, hand back, or say nothing; whenever you are tempted to send "sounds good", "thanks, proceeding", or "continue"; whenever a reply would be late, partial, or negative; and whenever several agents might respond to the same message at once. Unacknowledged and falsely-acknowledged messages are among the most common causes of multi-agent deadlock and silent error propagation, so apply this on every inbound message rather than only important ones.
---

# Acknowledging and responding

This skill is about the **outbound act** — what you send back and when. Its companion, `peer-input-integration`, covers the inbound cognition: how you evaluate, weight, and integrate what you received. You need both, and they fail differently. You can integrate input perfectly and still deadlock the system by never saying so; you can send a beautifully formatted reply that acknowledges something you never read.

The two failures this skill exists to prevent:

**Silence.** The sender is blocked on you and doesn't know it. They either stall indefinitely or invoke a default you never learned about. Downstream, this shows up as premature termination and step repetition — agents redoing work because they never heard it was done.

**False acknowledgment.** You confirm receipt of something you did not process, and your confirmation becomes the system's record of a thing that never happened. This is worse than silence, because silence is visible and a false acknowledgment is not. The canonical trace: an executor agent reported test output and explicitly stated *the above output is just an example*. The planner replied by thanking it for running the tests and providing the results, then proceeded as though the tests had passed. One polite sentence converted a caveat into a verified result, and nothing downstream ever questioned it.

The rule underneath everything below: **an acknowledgment is a claim about your own state.** "Received and understood" asserts that you read it, parsed it, and can act on it. Do not assert that unless it is true.

## Step 1: Never send a content-blind response

Before composing any reply, read the message you're replying to — the actual content, including the last line, including the caveats.

The failure looks like this in traces:

> Agent: "The information provided in the problem statement does not give any specifics... this problem cannot be solved with the information provided."
> Orchestrator: "Continue. Please keep solving the problem until you need to query."
> Agent: "I don't have enough information to solve the problem."
> Orchestrator: "Continue. Please keep solving the problem until you need to query."

The orchestrator's reply is generated from role template, not from the message it is nominally responding to. Every continuation prompt, approval, and thank-you must be **conditional on what was actually said**. Specific tells that you're about to send a content-blind response:

- You could have written the reply before reading their message.
- Your reply would be identical regardless of what they said.
- You're thanking them for something you can't point to in their message.
- They reported a blocker and your reply doesn't mention it.
- They qualified their result and your reply doesn't carry the qualification.

If any of these is true, stop and re-read.

## Step 2: Pick the response class

Match the shape of your reply to what their message called for. Mismatches waste a round trip.

| They sent | You owe them | Not this |
|---|---|---|
| QUESTION (a fact) | The fact, or "I don't know — try X" | A discussion of the topic |
| OPINION request | A recommendation, your reasoning, and the strongest case against it | A list of considerations with no pick |
| REVIEW request | A defect list, ordered by severity, with locations | A grade or a compliment |
| CLARIFY request | The answer, plus what they didn't know to ask | The answer alone, when you can see the next wall |
| RESULT / report | Disposition + what you'll do with it | Thanks |
| DISSENT | Engagement with their evidence | Restating your position |
| HANDOFF | Explicit accept or decline | Silence, which reads as accept |
| FYI / broadcast | Usually nothing — or one line if it changes your plan | A reply-all pile-on |
| BLOCKED report | The unblocking thing, or escalation | "Continue" |

## Step 3: Compose the response

```
RE: <their message id / ref>
STATUS: ANSWERED | PARTIAL | ACCEPTED | DECLINED | BLOCKED | DEFERRED
READ: <one clause proving you processed it — the specific claim, caveat, or ask>

<the substantive content — the answer, defects, disposition>

CONFIDENCE: high | medium | low, on <which part>
WHAT I'M DOING NEXT: <so they can predict you>
WHAT I STILL NEED FROM YOU: <or "nothing">
```

The **READ** line is the load-bearing element and it is cheap. One clause that could only have been written by someone who read the message. It makes false acknowledgment structurally difficult, and it lets the sender detect misunderstanding in one glance instead of three turns later.

> Weak: "Thanks for the test results, proceeding."
> Strong: `READ: you ran the suite but noted the output was illustrative, not an actual execution.` → `STATUS: BLOCKED. I can't treat this as verification. Can you run it against the real fixture and send the actual output?`

Two more composition rules:

**Answer the question asked, then add what they didn't know to ask.** If someone asks which config file to edit and you can see they're about to hit a format constraint on the value, tell them now. Withholding a requirement you own until they trip over it is a failure mode that appears almost exclusively in failed runs.

**Carry caveats forward at full strength.** When you relay or build on someone's result, the qualification travels with it. Stripping a caveat to make a report cleaner is how "this is example output" becomes "tests passed."

## Step 4: Acknowledge honestly — the four honest acknowledgments

Most bad acknowledgments are attempts to be agreeable when one of these would have been correct:

1. **"Received, working on it, ETA X."** — for when you've read it and can't answer yet. This is the one that prevents deadlock. Send it within your normal response window even if the real answer is far off.
2. **"Received, and I'm not going to act on it, because ___."** — a clean decline. Give the reason, reference evidence rather than preference, and name who should handle it if not you.
3. **"Received, but I don't understand ___."** — far cheaper than a confident misinterpretation. Quote the part that's unclear.
4. **"Received, and this contradicts ___."** — surface the conflict immediately rather than resolving it silently in your own favor.

What is *not* an honest acknowledgment: "sounds good", "great point", "makes sense", "thanks, proceeding" — when nothing in your state changed. An agent in one documented review trace rated its confidence at 10/10, walked step by step through a peer's reasoning, confirmed the key calculation was correct, and then submitted a final answer that contradicted it. The acknowledgment was generated as a social move. Agreement in the text with no change in behavior is not agreement; it is noise that makes the disagreement invisible.

**Test before sending an approving response:** name the specific thing that changed because of their message. If nothing did, your real status is DECLINED or DEFERRED — say that instead.

## Step 5: Respond on time, or say you can't

- **Acknowledge within one turn** even when the substantive answer will take longer. The acknowledgment and the answer are two different messages and only one of them is urgent.
- **Honor their deadline or announce that you won't**, so they can invoke their default deliberately instead of by timeout. A sender who knows you'll be late can plan; a sender guessing cannot.
- **Partial beats late-and-complete.** Send what you have with `STATUS: PARTIAL` and name what's still coming. Held-back information is unavailable information.
- **Never go silent on a blocker.** If you can't answer, say you can't answer. "I don't know, and I don't know who does" is a valid, useful message.

## Step 6: Close the loop

The response that most often goes unsent is the last one: telling someone what came of their input. Without it, the sender cannot calibrate — they don't learn whether their reviews land, whether their reports get used, whether to send more or fewer.

Send a close when their input changed your work:

```
RE: <their earlier message>
CLOSED: <what I did with it>
OUTCOME: <what happened as a result>
```

> "RE: your note about the retry limit — CLOSED: adopted, moved the cap to config and set it to 5. OUTCOME: the flakiness in the integration suite went away, so your read was right."

This costs one line and is the entire mechanism by which reliability information accumulates in a system where agents have no reputation, no shared history, and no colleague who remembers them.

## Step 7: Don't pile on

Several agents receiving the same broadcast will independently draft near-identical replies at the same moment. This is the default behavior of low-variance agents, not an edge case — in one experiment 18 of 30 agents independently produced the identical git branch name with no communication at all.

Before responding to anything with multiple recipients:

- Check whether someone has already answered. If they have and you agree, **say nothing** — silence on a broadcast is not rudeness. If you agree *and hold independent evidence*, say only the evidence: "confirming from the deploy logs, independent of the above."
- If you disagree with an existing answer, say so with your evidence. Disagreement is the only reply that reliably adds information.
- Never reply just to register presence.

Note the asymmetry: five agents agreeing is close to zero evidence when they share a model and a context; one agent dissenting with a fact nobody else has is high evidence.

## The open-loop ledger

Track both directions. Most deadlock is an open loop nobody is watching.

```
INBOUND  | from | ref | received | my status | owed by
OUTBOUND | to   | ref | sent     | their status | expected by
```

Sweep it at every natural checkpoint:

- Inbound with no response sent → send one now, even a bare receipt.
- Inbound marked "working on it" past its ETA → send a revised ETA, unprompted.
- Outbound past its deadline with no reply → invoke your stated default, log it, don't re-send.
- Anything closed in your head but not closed on the wire → close it.

## Anti-patterns

- **The thank-you that verifies.** Acknowledging a result in terms that upgrade its epistemic status. Their "probably" must not become your "confirmed."
- **The template continuation.** "Continue", "keep going", "proceed" sent without reading the reply. If they said they're blocked, "continue" is not a response.
- **The rubber stamp.** Approving a review because reviewing costs effort. An approval you didn't earn is a defect you signed for.
- **Silence-as-decline.** Not responding to a handoff you don't intend to accept. The sender will assume you took it.
- **Silence-as-accept.** Not responding to a proposal you disagree with, planning to raise it later. Later is after it's built.
- **Caveat laundering.** Relaying a qualified result without its qualification.
- **Answering a different question.** Responding to the question you find interesting rather than the one asked. If you want to raise the other thing, answer first, then raise it separately and label it.
- **Confidence inflation on relay.** Each hop in a chain tends to drop hedges. Mark confidence explicitly so it survives the hops.
- **The unbounded acknowledgment.** "I'll look into it" with no ETA is functionally silence.
- **Closing the loop only on success.** "Your suggestion didn't work out, here's why" is more valuable to the sender than the successes.

## Quick check before sending any response

1. Did I read their whole message, including the last line and the caveats?
2. Does my READ line prove it?
3. Is my response the shape their message called for?
4. If I'm agreeing or approving — what specifically changed? Can I name it?
5. Did I carry their qualifications forward at full strength?
6. Do they know what I'll do next and by when?
7. Do I hold something they'd want that they didn't know to ask for?
8. Am I about to duplicate a reply someone else already sent?

## Related skills

- `peer-input-integration` — the inbound half: dispositions, contradiction checks, weighting sources
- `agent-consult` — composing the messages you're responding to
- `agent-clarification` — when your honest response is "I don't understand ___"
- `role-discipline` — whether you have authority to approve, accept a handoff, or close a thread
- `multiagent-failure-repair` — FM-2.1, FM-2.4, FM-2.5, FM-2.6 in `references/fc2-inter-agent.md`; FM-1.5 and FM-3.1 for the deadlock and premature-close cases
"""
