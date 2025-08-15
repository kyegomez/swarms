"""
EuroSwarm Parliament - European Parliament Simulation

A comprehensive simulation of the European Parliament with 717 MEPs (Members of European Parliament)
based on real EU data, featuring full democratic functionality including bill introduction, committee work,
parliamentary debates, and democratic voting mechanisms.

Enhanced with hierarchical democratic structure where each political group operates as a specialized
Board of Directors with expertise areas, and a Parliament Speaker aggregates decisions using weighted voting.

Includes Wikipedia personality system for realistic, personality-driven MEP behavior based on real biographical data.
"""

from euroswarm_parliament import (
    EuroSwarmParliament,
    ParliamentaryMember,
    ParliamentaryBill,
    ParliamentaryVote,
    ParliamentaryCommittee,
    PoliticalGroupBoard,
    ParliamentSpeaker,
    ParliamentaryRole,
    VoteType,
    VoteResult,
)

# Import Wikipedia personality system
try:
    from wikipedia_personality_scraper import (
        WikipediaPersonalityScraper,
        MEPPersonalityProfile,
    )
    WIKIPEDIA_PERSONALITY_AVAILABLE = True
except ImportError:
    WIKIPEDIA_PERSONALITY_AVAILABLE = False

__version__ = "2.1.0"
__author__ = "Swarms Democracy Team"
__description__ = "European Parliament Simulation with Enhanced Hierarchical Democratic Functionality and Wikipedia Personality System"

__all__ = [
    "EuroSwarmParliament",
    "ParliamentaryMember",
    "ParliamentaryBill",
    "ParliamentaryVote",
    "ParliamentaryCommittee",
    "PoliticalGroupBoard",
    "ParliamentSpeaker",
    "ParliamentaryRole",
    "VoteType",
    "VoteResult",
    "WikipediaPersonalityScraper",
    "MEPPersonalityProfile",
    "WIKIPEDIA_PERSONALITY_AVAILABLE",
] 