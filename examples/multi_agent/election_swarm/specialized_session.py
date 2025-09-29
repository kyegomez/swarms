"""
ElectionSwarm Specialized Session Example
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from swarms.structs.election_swarm import ElectionSwarm, ElectionAlgorithm

# Create election
election = ElectionSwarm(verbose=True)

# Run specialized session
session_result = election.run_election_session(
    election_type=ElectionAlgorithm.CONSENSUS,
    max_rounds=3
)