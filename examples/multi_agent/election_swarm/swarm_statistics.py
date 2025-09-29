"""
ElectionSwarm Statistics Example
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from swarms.structs.election_swarm import ElectionSwarm

# Create election
election = ElectionSwarm(verbose=True)

# Get statistics
stats = election.get_election_statistics()