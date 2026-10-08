# Alternate Debate Structures

Eight scripted multi-agent conversation patterns that used to live in
`swarms/structs/multi_agent_debates.py`.

They were moved here because they are **compositions, not framework machinery**:
each one is built entirely from the public `Agent`, `Conversation`, and
`history_output_formatter` APIs, adds no new capability to the framework, and is
most useful as something you copy and adapt to your own domain.

| File | Structure | Shape |
|---|---|---|
| `interview_series.py` | `InterviewSeries` | interviewer + interviewee, prepared questions with follow-ups |
| `mentorship_session.py` | `MentorshipSession` | mentor + mentee, repeated sessions with optional feedback |
| `peer_review_process.py` | `PeerReviewProcess` | N reviewers + author, review/response rounds |
| `mediation_session.py` | `MediationSession` | N parties + mediator, working toward resolution |
| `brainstorming_session.py` | `BrainstormingSession` | N participants + facilitator, ideas build on each other |
| `council_meeting.py` | `CouncilMeeting` | N members + chairperson, discussion then a vote |
| `negotiation_session.py` | `NegotiationSession` | N parties + mediator, positions, responses, concessions |
| `trial_simulation.py` | `TrialSimulation` | prosecution / defense / judge / witnesses across trial phases |

Each file is self-contained: the class plus a runnable demo. Run any of them
directly from the repo root:

```bash
python examples/multi_agent/alternate_debates/council_meeting.py
```

## Using one in your own code

Copy the file into your project and edit the agents and prompts — that is the
intended path. If you would rather import it from here, add this folder to the
path first:

```python
import sys
from pathlib import Path

sys.path.insert(0, "examples/multi_agent/alternate_debates")

from council_meeting import CouncilMeeting
```

The examples in `../orchestration_examples/` do exactly this.

## The shared template

Five of these eight (`PeerReviewProcess`, `MediationSession`,
`BrainstormingSession`, `CouncilMeeting`, `NegotiationSession`), along with
`ExpertPanelDiscussion` still in the library, follow one pattern:

1. Validate that there are at least two participants and a leader
2. Broadcast an intro to the leader naming every participant
3. Broadcast an intro to each participant naming the others
4. For each round: the leader opens, each participant responds, the leader
   synthesizes the last `len(participants)` messages
5. Return `history_output_formatter(conversation, output_type)`

If you are writing a new one, start from whichever of these is closest and
change the role nouns and prompts.

`InterviewSeries` and `MentorshipSession` are two-agent variants of the same
idea. `TrialSimulation` is genuinely different — a phase state machine rather
than a symmetric participant loop.

## Known issue in `trial_simulation.py`

The `cross` phase reads `witness_testimony`, a loop variable left over from the
`testimony` phase. As a result every witness is cross-examined on the **last**
witness's testimony, and passing a `phases` list that omits `"testimony"` raises
`NameError`. Keep `"testimony"` before `"cross"` until this is fixed.

## Tests

`test_alternate_debates.py` holds the tests that moved with these structures:

```bash
pytest examples/multi_agent/alternate_debates/test_alternate_debates.py
```
