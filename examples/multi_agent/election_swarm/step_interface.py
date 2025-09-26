"""
ElectionSwarm Step Interface Example
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from swarms.structs.election_swarm import ElectionSwarm, ElectionAlgorithm

# Create election
election = ElectionSwarm(verbose=True)

# Run single step
task = "Vote on the best candidate for mayor based on their policies and experience"
step_result = election.step(task=task, election_type=ElectionAlgorithm.LEADER_ELECTION)