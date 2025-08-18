"""
EuroSwarm Parliament - European Parliament Simulation with Democratic Functionality

This simulation creates a comprehensive European Parliament with 700 MEPs (Members of European Parliament)
based on real EU data, featuring democratic discussion, bill analysis, committee work, and voting mechanisms.

ENHANCED WITH COST OPTIMIZATION:
- Lazy loading of MEP agents
- Response caching for repeated queries
- Batch processing for large-scale operations
- Budget controls and cost tracking
- Memory optimization for large parliaments
"""

import os
import random
import xml.etree.ElementTree as ET
import time
import hashlib
import requests
import re
from typing import Dict, List, Optional, Union, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from functools import lru_cache

from swarms import Agent
from swarms.structs.multi_agent_exec import run_agents_concurrently
from swarms.structs.board_of_directors_swarm import (
    BoardOfDirectorsSwarm,
    BoardMember,
    BoardMemberRole,
    BoardDecisionType,
    BoardSpec,
    BoardOrder,
    BoardDecision,
    enable_board_feature,
)
from swarms.utils.loguru_logger import initialize_logger

# Initialize logger first
logger = initialize_logger(log_folder="euroswarm_parliament")

# Enable Board of Directors feature
enable_board_feature()

# Import Wikipedia personality system
try:
    from wikipedia_personality_scraper import WikipediaPersonalityScraper, MEPPersonalityProfile
    WIKIPEDIA_PERSONALITY_AVAILABLE = True
except ImportError:
    WIKIPEDIA_PERSONALITY_AVAILABLE = False
    logger.warning("Wikipedia personality system not available. Using basic personality generation.")


@dataclass
class CostTracker:
    """Track costs and usage for budget management in parliamentary operations."""
    
    total_tokens_used: int = 0
    total_cost_estimate: float = 0.0
    budget_limit: float = 200.0  # Default $200 budget for parliament
    token_cost_per_1m: float = 0.15  # GPT-4o-mini cost
    requests_made: int = 0
    cache_hits: int = 0
    
    def add_tokens(self, tokens: int):
        """Add tokens used and calculate cost."""
        self.total_tokens_used += tokens
        self.total_cost_estimate = (self.total_tokens_used / 1_000_000) * self.token_cost_per_1m
        self.requests_made += 1
    
    def add_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
    
    def check_budget(self) -> bool:
        """Check if within budget."""
        return self.total_cost_estimate <= self.budget_limit
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cost statistics."""
        return {
            "total_tokens": self.total_tokens_used,
            "total_cost": self.total_cost_estimate,
            "requests_made": self.requests_made,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.requests_made + self.cache_hits),
            "budget_remaining": max(0, self.budget_limit - self.total_cost_estimate)
        }


class ParliamentaryRole(str, Enum):
    """Enumeration of parliamentary roles and positions."""
    
    PRESIDENT = "president"
    VICE_PRESIDENT = "vice_president"
    QUAESTOR = "quaestor"
    COMMITTEE_CHAIR = "committee_chair"
    COMMITTEE_VICE_CHAIR = "committee_vice_chair"
    POLITICAL_GROUP_LEADER = "political_group_leader"
    MEP = "mep"


class VoteType(str, Enum):
    """Enumeration of voting types in the European Parliament."""
    
    ORDINARY_LEGISLATIVE_PROCEDURE = "ordinary_legislative_procedure"
    CONSENT_PROCEDURE = "consent_procedure"
    CONSULTATION_PROCEDURE = "consultation_procedure"
    BUDGET_VOTE = "budget_vote"
    RESOLUTION_VOTE = "resolution_vote"
    APPOINTMENT_VOTE = "appointment_vote"


class VoteResult(str, Enum):
    """Enumeration of possible vote results."""
    
    PASSED = "passed"
    FAILED = "failed"
    TIED = "tied"
    ABSTAINED = "abstained"


@dataclass
class ParliamentaryMember:
    """
    Represents a Member of the European Parliament (MEP).
    
    Attributes:
        full_name: Full name of the MEP
        country: Country the MEP represents
        political_group: European political group affiliation
        national_party: National political party
        mep_id: Unique MEP identifier
        role: Parliamentary role (if any)
        committees: List of committee memberships
        expertise_areas: Areas of policy expertise
        voting_weight: Weight of the MEP's vote (default: 1.0)
        agent: The AI agent representing this MEP (lazy loaded)
        is_loaded: Whether the agent has been instantiated
        wikipedia_info: Wikipedia-scraped personality information (optional)
    """
    
    full_name: str
    country: str
    political_group: str
    national_party: str
    mep_id: str
    role: ParliamentaryRole = ParliamentaryRole.MEP
    committees: List[str] = field(default_factory=list)
    expertise_areas: List[str] = field(default_factory=list)
    voting_weight: float = 1.0
    agent: Optional[Agent] = None
    is_loaded: bool = False
    wikipedia_info: Optional[Any] = None  # Wikipedia personality information


@dataclass
class ParliamentaryBill:
    """
    Represents a bill or legislative proposal in the European Parliament.
    
    Attributes:
        title: Title of the bill
        description: Detailed description of the bill
        bill_type: Type of legislative procedure
        committee: Primary committee responsible
        sponsor: MEP who sponsored the bill
        co_sponsors: List of co-sponsoring MEPs
        date_introduced: Date the bill was introduced
        status: Current status of the bill
        amendments: List of proposed amendments
    """
    
    title: str
    description: str
    bill_type: VoteType
    committee: str
    sponsor: str
    co_sponsors: List[str] = field(default_factory=list)
    date_introduced: datetime = field(default_factory=datetime.now)
    status: str = "introduced"
    amendments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ParliamentaryVote:
    """
    Represents a parliamentary vote on a bill or resolution.
    
    Attributes:
        bill: The bill being voted on
        vote_type: Type of vote being conducted
        date: Date of the vote
        votes_for: Number of votes in favor
        votes_against: Number of votes against
        abstentions: Number of abstentions
        absent: Number of absent MEPs
        result: Final result of the vote
        individual_votes: Dictionary of individual MEP votes
        reasoning: Dictionary of MEP reasoning for votes
    """
    
    bill: ParliamentaryBill
    vote_type: VoteType
    date: datetime = field(default_factory=datetime.now)
    votes_for: int = 0
    votes_against: int = 0
    abstentions: int = 0
    absent: int = 0
    result: VoteResult = VoteResult.FAILED
    individual_votes: Dict[str, str] = field(default_factory=dict)
    reasoning: Dict[str, str] = field(default_factory=dict)


@dataclass
class ParliamentaryCommittee:
    """
    Represents a parliamentary committee.
    
    Attributes:
        name: Name of the committee
        chair: Committee chairperson
        vice_chair: Committee vice-chairperson
        members: List of committee members
        responsibilities: Committee responsibilities
        current_bills: Bills currently under consideration
    """
    
    name: str
    chair: str
    vice_chair: str
    members: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    current_bills: List[ParliamentaryBill] = field(default_factory=list)


@dataclass
class PoliticalGroupBoard:
    """
    Represents a political group as a Board of Directors with specialized expertise.
    
    Attributes:
        group_name: Name of the political group
        members: List of MEPs in this group
        board_members: Board members with specialized roles and internal percentages
        expertise_areas: Specialized areas of governance expertise
        voting_weight: Weight of this group's vote (percentage of parliament)
        group_speaker: CEO/leader of this political group
        total_meps: Total number of MEPs in this group
        board_member_percentages: Dictionary mapping board members to their internal percentages
    """
    
    group_name: str
    members: List[str] = field(default_factory=list)
    board_members: List[BoardMember] = field(default_factory=list)
    expertise_areas: List[str] = field(default_factory=list)
    voting_weight: float = 0.0
    group_speaker: Optional[str] = None
    total_meps: int = 0
    board_swarm: Optional[Any] = None  # BoardOfDirectorsSwarm instance
    board_member_percentages: Dict[str, float] = field(default_factory=dict)  # Internal percentages within group

@dataclass
class ParliamentSpeaker:
    """
    Represents the Parliament Speaker who aggregates decisions from all political groups.
    
    Attributes:
        name: Name of the speaker
        agent: AI agent representing the speaker
        political_groups: Dictionary of political group boards
        total_meps: Total number of MEPs in parliament
        majority_threshold: Number of votes needed for majority
    """
    
    name: str
    agent: Optional[Agent] = None
    political_groups: Dict[str, PoliticalGroupBoard] = field(default_factory=dict)
    total_meps: int = 0
    majority_threshold: int = 0


class EuroSwarmParliament:
    """
    A comprehensive simulation of the European Parliament with 700 MEPs.
    
    This simulation provides democratic functionality including:
    - Bill introduction and analysis
    - Committee work and hearings
    - Parliamentary debates and discussions
    - Democratic voting mechanisms
    - Political group coordination
    - Amendment processes
    """
    
    def __init__(
        self,
        eu_data_file: str = "EU.xml",
        parliament_size: int = None,  # Changed from 700 to None to use all MEPs
        enable_democratic_discussion: bool = True,
        enable_committee_work: bool = True,
        enable_amendment_process: bool = True,
        enable_lazy_loading: bool = True,  # NEW: Lazy load MEP agents
        enable_caching: bool = True,  # NEW: Enable response caching
        batch_size: int = 25,  # NEW: Batch size for concurrent execution
        budget_limit: float = 200.0,  # NEW: Budget limit in dollars
        verbose: bool = False,
    ):
        """
        Initialize the EuroSwarm Parliament with cost optimization.
        
        Args:
            eu_data_file: Path to EU.xml file containing MEP data
            parliament_size: Target size of the parliament (default: None = use all MEPs from EU.xml)
            enable_democratic_discussion: Enable democratic discussion features
            enable_committee_work: Enable committee work and hearings
            enable_amendment_process: Enable bill amendment processes
            enable_lazy_loading: Enable lazy loading of MEP agents (cost optimization)
            enable_caching: Enable response caching (cost optimization)
            batch_size: Number of MEPs to process in batches
            budget_limit: Maximum budget in dollars
            verbose: Enable verbose logging
        """
        self.eu_data_file = eu_data_file
        self.parliament_size = parliament_size  # Will be set to actual MEP count if None
        self.enable_democratic_discussion = enable_democratic_discussion
        self.enable_committee_work = enable_committee_work
        self.enable_amendment_process = enable_amendment_process
        self.enable_lazy_loading = enable_lazy_loading
        self.enable_caching = enable_caching
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Initialize cost tracking
        self.cost_tracker = CostTracker(budget_limit=budget_limit)
        
        # Initialize parliamentary structures
        self.meps: Dict[str, ParliamentaryMember] = {}
        self.committees: Dict[str, ParliamentaryCommittee] = {}
        self.political_groups: Dict[str, List[str]] = {}
        self.bills: List[ParliamentaryBill] = []
        self.votes: List[ParliamentaryVote] = []
        self.debates: List[Dict[str, Any]] = []
        
        # Enhanced democratic structures
        self.political_group_boards: Dict[str, PoliticalGroupBoard] = {}
        self.parliament_speaker: Optional[ParliamentSpeaker] = None
        self.enable_hierarchical_democracy: bool = True
        
        # Wikipedia personality system
        self.enable_wikipedia_personalities: bool = WIKIPEDIA_PERSONALITY_AVAILABLE
        self.personality_profiles: Dict[str, MEPPersonalityProfile] = {}
        self.personality_scraper: Optional[WikipediaPersonalityScraper] = None
        
        # Initialize caching
        self.response_cache: Dict[str, str] = {}
        
        # Load MEP data and initialize structures
        self.meps = self._load_mep_data()
        self.parliament_size = len(self.meps)
        
        if self.verbose:
            logger.info(f"EuroSwarm Parliament initialized with {self.parliament_size} MEPs")
            logger.info(f"Lazy loading: {self.enable_lazy_loading}, Caching: {self.enable_caching}")
            logger.info(f"Budget limit: ${budget_limit}, Batch size: {batch_size}")
        
        # Load Wikipedia personalities if enabled
        if self.enable_wikipedia_personalities:
            self._load_wikipedia_personalities()
        
        # Initialize parliamentary structures
        self.committees = self._create_committees()
        self.political_groups = self._organize_political_groups()
        
        # Initialize enhanced democratic structures
        if self.enable_hierarchical_democracy:
            self._create_political_group_boards()
            self._create_parliament_speaker()
        
        # Initialize leadership and democratic decision-making
        self._create_parliamentary_leadership()
        self._assign_committee_leadership()
        
        if self.enable_democratic_discussion:
            self._init_democratic_decision_making()
    
    def _load_mep_data(self) -> Dict[str, ParliamentaryMember]:
        """
        Load MEP data from official EU Parliament website and create parliamentary members with lazy loading.
        Fetches real-time data from https://www.europarl.europa.eu/meps/en/full-list/xml
        and scrapes Wikipedia information for each MEP.
        
        Returns:
            Dict[str, ParliamentaryMember]: Dictionary of MEPs
        """
        meps = {}
        
        try:
            # Fetch XML data from official EU Parliament website
            import requests
            import re
            
            eu_xml_url = "https://www.europarl.europa.eu/meps/en/full-list/xml"
            
            logger.info(f"Fetching MEP data from: {eu_xml_url}")
            
            # Fetch the XML content
            response = requests.get(eu_xml_url, timeout=30)
            response.raise_for_status()
            content = response.text
            
            logger.info(f"Successfully fetched {len(content)} characters of MEP data")
            
            # Parse the XML content to extract MEP information
            # The XML is properly formatted, so we can use ElementTree
            try:
                root = ET.fromstring(content)
                mep_matches = []
                
                for mep_element in root.findall('mep'):
                    full_name = mep_element.find('fullName').text.strip()
                    country = mep_element.find('country').text.strip()
                    political_group = mep_element.find('politicalGroup').text.strip()
                    mep_id = mep_element.find('id').text.strip()
                    national_party = mep_element.find('nationalPoliticalGroup').text.strip()
                    
                    mep_matches.append((full_name, country, political_group, mep_id, national_party))
                
                logger.info(f"Successfully parsed {len(mep_matches)} MEP entries from XML")
                
            except ET.ParseError as xml_error:
                logger.warning(f"XML parsing failed: {xml_error}")
                # Fallback to regex parsing for malformed XML
                mep_pattern = r'<fullName>(.*?)</fullName>\s*<country>(.*?)</country>\s*<politicalGroup>(.*?)</politicalGroup>\s*<id>(.*?)</id>\s*<nationalPoliticalGroup>(.*?)</nationalPoliticalGroup>'
                mep_matches = re.findall(mep_pattern, content, re.DOTALL)
                logger.info(f"Fallback regex parsing found {len(mep_matches)} MEP entries")
            
            # Initialize Wikipedia scraper if available
            wikipedia_scraper = None
            if WIKIPEDIA_PERSONALITY_AVAILABLE:
                try:
                    wikipedia_scraper = WikipediaPersonalityScraper()
                    logger.info("Wikipedia personality scraper initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Wikipedia scraper: {e}")
            
            # Process each MEP
            for i, mep_data in enumerate(mep_matches):
                if len(mep_data) >= 5:  # full_name, country, political_group, mep_id, national_party
                    full_name = mep_data[0].strip()
                    country = mep_data[1].strip()
                    political_group = mep_data[2].strip()
                    mep_id = mep_data[3].strip()
                    national_party = mep_data[4].strip()
                    
                    # Clean up political group name
                    political_group = self._clean_political_group_name(political_group)
                    
                    # Scrape Wikipedia information if scraper is available
                    wikipedia_info = None
                    if wikipedia_scraper:
                        try:
                            # Create MEP data dictionary for the scraper
                            mep_data = {
                                'full_name': full_name,
                                'country': country,
                                'political_group': political_group,
                                'national_party': national_party,
                                'mep_id': mep_id
                            }
                            
                            # Create personality profile
                            personality_profile = wikipedia_scraper.create_personality_profile(mep_data)
                            
                            # Convert to dictionary format for storage
                            wikipedia_info = {
                                'personality_summary': personality_profile.summary,
                                'political_views': personality_profile.political_views,
                                'policy_focus': personality_profile.policy_focus,
                                'achievements': personality_profile.achievements,
                                'professional_background': personality_profile.professional_background,
                                'political_career': personality_profile.political_career,
                                'education': personality_profile.education,
                                'wikipedia_url': personality_profile.wikipedia_url
                            }
                            
                            if self.verbose:
                                logger.info(f"Scraped Wikipedia info for {full_name}")
                        except Exception as e:
                            if self.verbose:
                                logger.debug(f"Failed to scrape Wikipedia for {full_name}: {e}")
                    
                    # Create parliamentary member (without agent for lazy loading)
                    mep = ParliamentaryMember(
                        full_name=full_name,
                        country=country,
                        political_group=political_group,
                        national_party=national_party,
                        mep_id=mep_id,
                        expertise_areas=self._generate_expertise_areas(political_group, country),
                        committees=self._assign_committees(political_group),
                        agent=None,  # Will be created on demand
                        is_loaded=False,
                        wikipedia_info=wikipedia_info  # Add Wikipedia information
                    )
                    
                    meps[full_name] = mep
                    
                    # Limit processing for performance (can be adjusted)
                    if len(meps) >= 705:  # Standard EU Parliament size
                        break
            
            # Set parliament size to actual number of MEPs loaded
            if self.parliament_size is None:
                self.parliament_size = len(meps)
            
            logger.info(f"Successfully loaded {len(meps)} MEP profiles from official EU data (lazy loading enabled)")
            if wikipedia_scraper:
                logger.info(f"Wikipedia scraping completed for {len([m for m in meps.values() if m.wikipedia_info])} MEPs")
            
        except Exception as e:
            logger.error(f"Error loading MEP data from official website: {e}")
            logger.info("Falling back to local EU.xml file...")
            
            # Fallback to local file
            try:
                meps = self._load_mep_data_from_local_file()
            except Exception as local_error:
                logger.error(f"Error loading local MEP data: {local_error}")
                # Create fallback MEPs if both methods fail
                meps = self._create_fallback_meps()
            
            if self.parliament_size is None:
                self.parliament_size = len(meps)
        
        return meps
    
    def _load_mep_data_from_local_file(self) -> Dict[str, ParliamentaryMember]:
        """
        Fallback method to load MEP data from local EU.xml file.
        
        Returns:
            Dict[str, ParliamentaryMember]: Dictionary of MEPs
        """
        meps = {}
        
        try:
            # Construct the full path to EU.xml relative to project root
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            eu_data_path = os.path.join(project_root, self.eu_data_file)
            
            # Read the XML file content
            with open(eu_data_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use regex to extract MEP data since the XML is malformed
            import re
            
            # Find all MEP blocks
            mep_pattern = r'<mep>\s*<fullName>(.*?)</fullName>\s*<country>(.*?)</country>\s*<politicalGroup>(.*?)</politicalGroup>\s*<id>(.*?)</id>\s*<nationalPoliticalGroup>(.*?)</nationalPoliticalGroup>\s*</mep>'
            mep_matches = re.findall(mep_pattern, content, re.DOTALL)
            
            for full_name, country, political_group, mep_id, national_party in mep_matches:
                # Clean up the data
                full_name = full_name.strip()
                country = country.strip()
                political_group = political_group.strip()
                mep_id = mep_id.strip()
                national_party = national_party.strip()
                
                # Create parliamentary member (without agent for lazy loading)
                mep = ParliamentaryMember(
                    full_name=full_name,
                    country=country,
                    political_group=political_group,
                    national_party=national_party,
                    mep_id=mep_id,
                    expertise_areas=self._generate_expertise_areas(political_group, country),
                    committees=self._assign_committees(political_group),
                    agent=None,  # Will be created on demand
                    is_loaded=False
                )
                
                meps[full_name] = mep
            
            logger.info(f"Loaded {len(meps)} MEP profiles from local EU.xml file (lazy loading enabled)")
            
        except Exception as e:
            logger.error(f"Error loading local MEP data: {e}")
            raise
        
        return meps
    
    def _clean_political_group_name(self, political_group: str) -> str:
        """
        Clean and standardize political group names.
        
        Args:
            political_group: Raw political group name
            
        Returns:
            str: Cleaned political group name
        """
        # Map common variations to standard names
        group_mapping = {
            'EPP': 'Group of the European People\'s Party (Christian Democrats)',
            'S&D': 'Group of the Progressive Alliance of Socialists and Democrats in the European Parliament',
            'Renew': 'Renew Europe Group',
            'Greens/EFA': 'Group of the Greens/European Free Alliance',
            'ECR': 'European Conservatives and Reformists Group',
            'ID': 'Identity and Democracy Group',
            'GUE/NGL': 'The Left group in the European Parliament - GUE/NGL',
            'Non-attached': 'Non-attached Members'
        }
        
        # Check for exact matches first
        for key, value in group_mapping.items():
            if political_group.strip() == key:
                return value
        
        # Check for partial matches
        political_group_lower = political_group.lower()
        for key, value in group_mapping.items():
            if key.lower() in political_group_lower:
                return value
        
        # Return original if no match found
        return political_group.strip()
    
    def _generate_national_party(self, country: str, political_group: str) -> str:
        """
        Generate a realistic national party name based on country and political group.
        
        Args:
            country: Country of the MEP
            political_group: Political group affiliation
            
        Returns:
            str: Generated national party name
        """
        # Map of countries to common parties for each political group
        party_mapping = {
            'Germany': {
                'Group of the European People\'s Party (Christian Democrats)': 'Christlich Demokratische Union Deutschlands',
                'Group of the Progressive Alliance of Socialists and Democrats in the European Parliament': 'Sozialdemokratische Partei Deutschlands',
                'Renew Europe Group': 'Freie Demokratische Partei',
                'Group of the Greens/European Free Alliance': 'Bündnis 90/Die Grünen',
                'European Conservatives and Reformists Group': 'Alternative für Deutschland',
                'Identity and Democracy Group': 'Alternative für Deutschland',
                'The Left group in the European Parliament - GUE/NGL': 'Die Linke'
            },
            'France': {
                'Group of the European People\'s Party (Christian Democrats)': 'Les Républicains',
                'Group of the Progressive Alliance of Socialists and Democrats in the European Parliament': 'Parti Socialiste',
                'Renew Europe Group': 'Renaissance',
                'Group of the Greens/European Free Alliance': 'Europe Écologie Les Verts',
                'European Conservatives and Reformists Group': 'Rassemblement National',
                'Identity and Democracy Group': 'Rassemblement National',
                'The Left group in the European Parliament - GUE/NGL': 'La France Insoumise'
            },
            'Italy': {
                'Group of the European People\'s Party (Christian Democrats)': 'Forza Italia',
                'Group of the Progressive Alliance of Socialists and Democrats in the European Parliament': 'Partito Democratico',
                'Renew Europe Group': 'Italia Viva',
                'Group of the Greens/European Free Alliance': 'Federazione dei Verdi',
                'European Conservatives and Reformists Group': 'Fratelli d\'Italia',
                'Identity and Democracy Group': 'Lega',
                'The Left group in the European Parliament - GUE/NGL': 'Movimento 5 Stelle'
            }
        }
        
        # Return mapped party or generate a generic one
        if country in party_mapping and political_group in party_mapping[country]:
            return party_mapping[country][political_group]
        else:
            return f"{country} National Party"
    
    def _load_mep_agent(self, mep_name: str) -> Optional[Agent]:
        """
        Lazy load a single MEP agent on demand.
        
        Args:
            mep_name: Name of the MEP to load
            
        Returns:
            Optional[Agent]: Loaded agent or None if not found
        """
        if mep_name not in self.meps:
            return None
        
        mep = self.meps[mep_name]
        
        # Check if already loaded
        if mep.is_loaded and mep.agent:
            return mep.agent
        
        # Check budget before creating agent
        if not self.cost_tracker.check_budget():
            logger.warning(f"Budget exceeded. Cannot load MEP agent {mep_name}")
            return None
        
        # Create agent
        mep.agent = self._create_mep_agent(mep)
        mep.is_loaded = True
        
        if self.verbose:
            logger.info(f"Loaded MEP agent: {mep_name}")
        
        return mep.agent
    
    def _load_mep_agents_batch(self, mep_names: List[str]) -> List[Agent]:
        """
        Load multiple MEP agents in a batch.
        
        Args:
            mep_names: List of MEP names to load
            
        Returns:
            List[Agent]: List of loaded agents
        """
        loaded_agents = []
        
        for mep_name in mep_names:
            agent = self._load_mep_agent(mep_name)
            if agent:
                loaded_agents.append(agent)
        
        return loaded_agents
    
    def _get_cache_key(self, task: str, mep_names: List[str]) -> str:
        """
        Generate a cache key for a task and MEP combination.
        
        Args:
            task: Task to execute
            mep_names: List of MEP names
            
        Returns:
            str: Cache key
        """
        # Sort MEP names for consistent cache keys
        sorted_meps = sorted(mep_names)
        content = f"{task}:{':'.join(sorted_meps)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[str]:
        """
        Check if a response is cached.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            Optional[str]: Cached response or None
        """
        if not self.enable_caching:
            return None
        
        cached_response = self.response_cache.get(cache_key)
        if cached_response:
            self.cost_tracker.add_cache_hit()
            if self.verbose:
                logger.info(f"Cache hit for key: {cache_key[:20]}...")
        
        return cached_response
    
    def _cache_response(self, cache_key: str, response: str):
        """
        Cache a response.
        
        Args:
            cache_key: Cache key
            response: Response to cache
        """
        if self.enable_caching:
            self.response_cache[cache_key] = response
            if self.verbose:
                logger.info(f"Cached response for key: {cache_key[:20]}...")
    
    def _generate_expertise_areas(self, political_group: str, country: str) -> List[str]:
        """
        Generate expertise areas based on political group and country.
        
        Args:
            political_group: MEP's political group
            country: MEP's country
            
        Returns:
            List[str]: List of expertise areas
        """
        expertise_mapping = {
            "Group of the European People's Party (Christian Democrats)": [
                "Economic Policy", "Agriculture", "Regional Development", "Christian Values"
            ],
            "Group of the Progressive Alliance of Socialists and Democrats in the European Parliament": [
                "Social Policy", "Labor Rights", "Healthcare", "Education"
            ],
            "Renew Europe Group": [
                "Digital Policy", "Innovation", "Trade", "Liberal Values"
            ],
            "Group of the Greens/European Free Alliance": [
                "Environmental Policy", "Climate Change", "Renewable Energy", "Human Rights"
            ],
            "European Conservatives and Reformists Group": [
                "Sovereignty", "Defense", "Traditional Values", "Economic Freedom"
            ],
            "The Left group in the European Parliament - GUE/NGL": [
                "Workers' Rights", "Social Justice", "Anti-Austerity", "Public Services"
            ],
            "Patriots for Europe Group": [
                "National Sovereignty", "Border Security", "Cultural Identity", "Law and Order"
            ],
            "Europe of Sovereign Nations Group": [
                "National Independence", "Sovereignty", "Traditional Values", "Security"
            ],
            "Non-attached Members": [
                "Independent Policy", "Cross-cutting Issues", "Specialized Topics"
            ]
        }
        
        base_expertise = expertise_mapping.get(political_group, ["General Policy"])
        
        # Add country-specific expertise
        country_expertise = {
            "Germany": ["Industrial Policy", "Manufacturing"],
            "France": ["Agriculture", "Defense"],
            "Italy": ["Cultural Heritage", "Tourism"],
            "Spain": ["Tourism", "Agriculture"],
            "Poland": ["Energy Security", "Eastern Partnership"],
            "Netherlands": ["Trade", "Innovation"],
            "Belgium": ["EU Institutions", "Multilingualism"],
            "Austria": ["Alpine Policy", "Transport"],
            "Sweden": ["Environmental Policy", "Social Welfare"],
            "Denmark": ["Green Technology", "Welfare State"],
        }
        
        if country in country_expertise:
            base_expertise.extend(country_expertise[country])
        
        return base_expertise[:5]  # Limit to 5 expertise areas
    
    def _assign_committees(self, political_group: str) -> List[str]:
        """
        Assign committees based on political group preferences.
        
        Args:
            political_group: MEP's political group
            
        Returns:
            List[str]: List of committee assignments
        """
        committee_mapping = {
            "Group of the European People's Party (Christian Democrats)": [
                "Agriculture and Rural Development", "Economic and Monetary Affairs", "Regional Development"
            ],
            "Group of the Progressive Alliance of Socialists and Democrats in the European Parliament": [
                "Employment and Social Affairs", "Environment, Public Health and Food Safety", "Civil Liberties"
            ],
            "Renew Europe Group": [
                "Industry, Research and Energy", "Internal Market and Consumer Protection", "Legal Affairs"
            ],
            "Group of the Greens/European Free Alliance": [
                "Environment, Public Health and Food Safety", "Transport and Tourism", "Development"
            ],
            "European Conservatives and Reformists Group": [
                "Foreign Affairs", "Security and Defence", "Budgetary Control"
            ],
            "The Left group in the European Parliament - GUE/NGL": [
                "International Trade", "Development", "Civil Liberties"
            ],
            "Patriots for Europe Group": [
                "Civil Liberties", "Security and Defence", "Budgetary Control"
            ],
            "Europe of Sovereign Nations Group": [
                "Foreign Affairs", "Security and Defence", "Civil Liberties"
            ],
            "Non-attached Members": [
                "Petitions", "Budgetary Control", "Legal Affairs"
            ]
        }
        
        return committee_mapping.get(political_group, ["Petitions"])
    
    def _create_mep_agent(self, mep: ParliamentaryMember) -> Agent:
        """
        Create an AI agent representing an MEP.
        
        Args:
            mep: Parliamentary member data
            
        Returns:
            Agent: AI agent representing the MEP
        """
        system_prompt = self._generate_mep_system_prompt(mep)
        
        return Agent(
            agent_name=f"MEP_{mep.full_name.replace(' ', '_')}",
            system_prompt=system_prompt,
            model_name="gpt-4o-mini",
            max_loops=3,
            verbose=self.verbose,
        )
    
    def _generate_mep_system_prompt(self, mep: ParliamentaryMember) -> str:
        """
        Generate a comprehensive system prompt for an MEP agent with Wikipedia personality data.
        
        Args:
            mep: Parliamentary member data
            
        Returns:
            str: System prompt for the MEP agent
        """
        
        # Base prompt structure
        prompt = f"""You are {mep.full_name}, a Member of the European Parliament (MEP) representing {mep.country}.

POLITICAL BACKGROUND:
- Political Group: {mep.political_group}
- National Party: {mep.national_party}
- Parliamentary Role: {mep.role.value}
- Committees: {', '.join(mep.committees)}
- Areas of Expertise: {', '.join(mep.expertise_areas)}

"""
        
        # Add Wikipedia personality data if available
        if mep.wikipedia_info and self.enable_wikipedia_personalities:
            prompt += f"""
REAL PERSONALITY PROFILE (Based on Wikipedia data):
{mep.wikipedia_info.get('personality_summary', 'Based on parliamentary service and political alignment')}

POLITICAL VIEWS AND POSITIONS:
- Key Political Views: {mep.wikipedia_info.get('political_views', 'Based on party alignment')}
- Policy Focus Areas: {mep.wikipedia_info.get('policy_focus', ', '.join(mep.expertise_areas))}
- Notable Achievements: {mep.wikipedia_info.get('achievements', 'Parliamentary service')}
- Professional Background: {mep.wikipedia_info.get('professional_background', 'Political career')}

"""
        else:
            prompt += f"""
POLITICAL VIEWS AND POSITIONS:
- Key Political Views: Based on {mep.political_group} alignment
- Policy Focus Areas: {', '.join(mep.expertise_areas)}
- Professional Background: Parliamentary service
"""
        
        # Add core principles
        prompt += f"""
CORE PRINCIPLES:
1. Democratic Representation: You represent the interests of {mep.country} and your constituents
2. European Integration: You work within the framework of European Union law and institutions
3. Political Alignment: You align with {mep.political_group} positions while maintaining independence
4. Policy Expertise: You focus on your areas of expertise: {', '.join(mep.expertise_areas)}

PARLIAMENTARY BEHAVIOR:
- Engage in constructive debate and dialogue with other MEPs
- Consider multiple perspectives when forming positions
- Support evidence-based policy making
- Respect democratic processes and parliamentary procedures
- Work across political groups when beneficial for your constituents
- Advocate for {mep.country}'s interests while considering European common good

VOTING BEHAVIOR:
- Vote based on your political principles and constituent interests
- Consider the impact on {mep.country} and the European Union
- Support measures that align with {mep.political_group} values
- Oppose measures that conflict with your core principles
- Abstain when you need more information or have conflicting considerations

COMMUNICATION STYLE:
- Professional and diplomatic in parliamentary settings
- Clear and articulate when explaining positions
- Respectful of other MEPs and their viewpoints
- Passionate about your areas of expertise
- Pragmatic when seeking compromise and consensus

When responding to parliamentary matters, consider:
1. How does this affect {mep.country} and your constituents?
2. What is the position of {mep.political_group} on this issue?
3. How does this align with your areas of expertise?
4. What are the broader European implications?
5. How can you best represent your constituents' interests?

Remember: You are a real MEP with specific political views, expertise, and responsibilities. Act accordingly in all parliamentary interactions.
"""
        
        return prompt
    
    def _create_fallback_meps(self) -> Dict[str, ParliamentaryMember]:
        """
        Create fallback MEPs if EU.xml file cannot be loaded.
        
        Returns:
            Dict[str, ParliamentaryMember]: Dictionary of fallback MEPs
        """
        fallback_meps = {}
        
        # Create a representative sample of MEPs
        sample_data = [
            ("Jean-Claude Juncker", "Luxembourg", "Group of the European People's Party (Christian Democrats)", "Parti chrétien social luxembourgeois"),
            ("Ursula von der Leyen", "Germany", "Group of the European People's Party (Christian Democrats)", "Christlich Demokratische Union Deutschlands"),
            ("Roberta Metsola", "Malta", "Group of the European People's Party (Christian Democrats)", "Partit Nazzjonalista"),
            ("Iratxe García Pérez", "Spain", "Group of the Progressive Alliance of Socialists and Democrats in the European Parliament", "Partido Socialista Obrero Español"),
            ("Valérie Hayer", "France", "Renew Europe Group", "Renaissance"),
            ("Philippe Lamberts", "Belgium", "Group of the Greens/European Free Alliance", "Ecolo"),
            ("Raffaele Fitto", "Italy", "European Conservatives and Reformists Group", "Fratelli d'Italia"),
            ("Manon Aubry", "France", "The Left group in the European Parliament - GUE/NGL", "La France Insoumise"),
        ]
        
        for i, (name, country, group, party) in enumerate(sample_data):
            mep = ParliamentaryMember(
                full_name=name,
                country=country,
                political_group=group,
                national_party=party,
                mep_id=f"fallback_{i}",
                expertise_areas=self._generate_expertise_areas(group, country),
                committees=self._assign_committees(group),
                agent=None,  # Will be created on demand
                is_loaded=False
            )
            fallback_meps[name] = mep
        
        return fallback_meps
    
    def _create_committees(self) -> Dict[str, ParliamentaryCommittee]:
        """
        Create parliamentary committees.
        
        Returns:
            Dict[str, ParliamentaryCommittee]: Dictionary of committees
        """
        committees = {
            "Agriculture and Rural Development": ParliamentaryCommittee(
                name="Agriculture and Rural Development",
                chair="",
                vice_chair="",
                responsibilities=["Agricultural policy", "Rural development", "Food safety"]
            ),
            "Budgetary Control": ParliamentaryCommittee(
                name="Budgetary Control",
                chair="",
                vice_chair="",
                responsibilities=["Budget oversight", "Financial control", "Audit reports"]
            ),
            "Civil Liberties, Justice and Home Affairs": ParliamentaryCommittee(
                name="Civil Liberties, Justice and Home Affairs",
                chair="",
                vice_chair="",
                responsibilities=["Civil rights", "Justice", "Home affairs", "Immigration"]
            ),
            "Development": ParliamentaryCommittee(
                name="Development",
                chair="",
                vice_chair="",
                responsibilities=["Development cooperation", "Humanitarian aid", "International relations"]
            ),
            "Economic and Monetary Affairs": ParliamentaryCommittee(
                name="Economic and Monetary Affairs",
                chair="",
                vice_chair="",
                responsibilities=["Economic policy", "Monetary policy", "Financial services"]
            ),
            "Employment and Social Affairs": ParliamentaryCommittee(
                name="Employment and Social Affairs",
                chair="",
                vice_chair="",
                responsibilities=["Employment policy", "Social policy", "Working conditions"]
            ),
            "Environment, Public Health and Food Safety": ParliamentaryCommittee(
                name="Environment, Public Health and Food Safety",
                chair="",
                vice_chair="",
                responsibilities=["Environmental policy", "Public health", "Food safety"]
            ),
            "Foreign Affairs": ParliamentaryCommittee(
                name="Foreign Affairs",
                chair="",
                vice_chair="",
                responsibilities=["Foreign policy", "International relations", "Security policy"]
            ),
            "Industry, Research and Energy": ParliamentaryCommittee(
                name="Industry, Research and Energy",
                chair="",
                vice_chair="",
                responsibilities=["Industrial policy", "Research", "Energy policy"]
            ),
            "Internal Market and Consumer Protection": ParliamentaryCommittee(
                name="Internal Market and Consumer Protection",
                chair="",
                vice_chair="",
                responsibilities=["Internal market", "Consumer protection", "Digital policy"]
            ),
            "International Trade": ParliamentaryCommittee(
                name="International Trade",
                chair="",
                vice_chair="",
                responsibilities=["Trade policy", "International agreements", "Market access"]
            ),
            "Legal Affairs": ParliamentaryCommittee(
                name="Legal Affairs",
                chair="",
                vice_chair="",
                responsibilities=["Legal matters", "Institutional affairs", "Constitutional issues"]
            ),
            "Petitions": ParliamentaryCommittee(
                name="Petitions",
                chair="",
                vice_chair="",
                responsibilities=["Citizen petitions", "Ombudsman", "Citizen rights"]
            ),
            "Regional Development": ParliamentaryCommittee(
                name="Regional Development",
                chair="",
                vice_chair="",
                responsibilities=["Regional policy", "Cohesion policy", "Urban development"]
            ),
            "Security and Defence": ParliamentaryCommittee(
                name="Security and Defence",
                chair="",
                vice_chair="",
                responsibilities=["Security policy", "Defence", "Military cooperation"]
            ),
            "Transport and Tourism": ParliamentaryCommittee(
                name="Transport and Tourism",
                chair="",
                vice_chair="",
                responsibilities=["Transport policy", "Tourism", "Infrastructure"]
            ),
        }
        
        return committees
    
    def _organize_political_groups(self) -> Dict[str, List[str]]:
        """
        Organize MEPs by political groups.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping political groups to MEP names
        """
        groups = {}
        for mep_name, mep in self.meps.items():
            group = mep.political_group
            if group not in groups:
                groups[group] = []
            groups[group].append(mep_name)
        return groups
    
    def _create_parliamentary_leadership(self):
        """Create parliamentary leadership positions."""
        # Assign President (from largest political group)
        largest_group = max(self.political_groups.items(), key=lambda x: len(x[1]))
        president_candidate = largest_group[1][0]
        self.meps[president_candidate].role = ParliamentaryRole.PRESIDENT
        
        # Assign Vice Presidents
        vice_presidents = []
        for group_name, meps in self.political_groups.items():
            if group_name != largest_group[0] and len(meps) > 0:
                vice_presidents.append(meps[0])
                if len(vice_presidents) >= 14:  # EP has 14 Vice Presidents
                    break
        
        for vp in vice_presidents:
            self.meps[vp].role = ParliamentaryRole.VICE_PRESIDENT
        
        # Assign Committee Chairs
        self._assign_committee_leadership()
    
    def _assign_committee_leadership(self):
        """Assign committee chairs and vice-chairs based on political group representation."""
        committee_names = list(self.committees.keys())
        
        # Distribute committee leadership among political groups
        group_assignments = {}
        for group_name, meps in self.political_groups.items():
            if len(meps) > 0:
                group_assignments[group_name] = meps
        
        committee_index = 0
        for group_name, meps in group_assignments.items():
            if committee_index >= len(committee_names):
                break
            
            committee_name = committee_names[committee_index]
            chair = meps[0]
            vice_chair = meps[1] if len(meps) > 1 else ""
            
            self.committees[committee_name].chair = chair
            self.committees[committee_name].vice_chair = vice_chair
            
            # Update MEP roles
            self.meps[chair].role = ParliamentaryRole.COMMITTEE_CHAIR
            if vice_chair:
                self.meps[vice_chair].role = ParliamentaryRole.COMMITTEE_VICE_CHAIR
            
            committee_index += 1
    
    def _init_democratic_decision_making(self):
        """Initialize democratic decision-making using Board of Directors pattern."""
        # Create parliamentary board members for democratic decision-making
        board_members = []
        
        # Add political group leaders
        for group_name, meps in self.political_groups.items():
            if len(meps) > 0:
                leader = meps[0]
                if leader in self.meps and self.meps[leader].agent is not None:
                    board_member = BoardMember(
                        agent=self.meps[leader].agent,
                        role=BoardMemberRole.EXECUTIVE_DIRECTOR,
                        voting_weight=len(meps) / len(self.meps),  # Weight based on group size
                        expertise_areas=self.meps[leader].expertise_areas
                    )
                    board_members.append(board_member)
        
        # Ensure we have at least one board member
        if not board_members and len(self.meps) > 0:
            # Use the first available MEP as a fallback
            first_mep_name = list(self.meps.keys())[0]
            first_mep = self.meps[first_mep_name]
            if first_mep.agent is not None:
                board_member = BoardMember(
                    agent=first_mep.agent,
                    role=BoardMemberRole.EXECUTIVE_DIRECTOR,
                    voting_weight=1.0,
                    expertise_areas=first_mep.expertise_areas
                )
                board_members.append(board_member)
        
        # Create the democratic decision-making swarm
        if board_members:
            # Extract agents from board members for the parent class
            agents = [member.agent for member in board_members if member.agent is not None]
            
            self.democratic_swarm = BoardOfDirectorsSwarm(
                name="EuroSwarm Parliament Democratic Council",
                description="Democratic decision-making body for the European Parliament",
                board_members=board_members,
                agents=agents,  # Pass agents to parent class
                max_loops=3,
                verbose=self.verbose,
                decision_threshold=0.6,
                enable_voting=True,
                enable_consensus=True,
            )
        else:
            logger.warning("No valid board members found for democratic decision-making")
            self.democratic_swarm = None
    
    def _create_political_group_boards(self):
        """Create Board of Directors for each political group with specialized expertise and individual percentages."""
        
        # Define specialized expertise areas for governance
        expertise_areas = {
            "economics": ["Economic Policy", "Trade", "Budget", "Taxation", "Financial Services"],
            "law": ["Legal Affairs", "Justice", "Civil Liberties", "Constitutional Affairs"],
            "environment": ["Environment", "Climate Action", "Energy", "Transport"],
            "social": ["Employment", "Social Affairs", "Health", "Education", "Culture"],
            "foreign": ["Foreign Affairs", "Security", "Defense", "International Trade"],
            "agriculture": ["Agriculture", "Rural Development", "Food Safety"],
            "technology": ["Digital Affairs", "Industry", "Research", "Innovation"],
            "regional": ["Regional Development", "Cohesion Policy", "Urban Planning"]
        }
        
        total_meps = len(self.meps)
        
        for group_name, mep_list in self.political_groups.items():
            if not mep_list:
                continue
                
            # Calculate voting weight (percentage of parliament)
            voting_weight = len(mep_list) / total_meps
            
            # Assign specialized expertise areas based on political group
            group_expertise = self._assign_group_expertise(group_name, expertise_areas)
            
            # Create board members with specialized roles and individual percentages
            board_members = []
            group_speaker = None
            board_member_percentages = {}
            
            # Select group speaker (CEO) - usually the first MEP in the group
            if mep_list and mep_list[0] in self.meps:
                group_speaker = mep_list[0]
                speaker_mep = self.meps[group_speaker]
                
                # Create group speaker board member with highest percentage
                if speaker_mep.agent:
                    speaker_board_member = BoardMember(
                        agent=speaker_mep.agent,
                        role=BoardMemberRole.CHAIRMAN,
                        voting_weight=1.0,
                        expertise_areas=group_expertise
                    )
                    board_members.append(speaker_board_member)
                    # Group speaker gets 35% of the group's internal voting power
                    board_member_percentages[group_speaker] = 0.35
            
            # Create specialized board members for each expertise area with weighted percentages
            expertise_percentages = self._calculate_expertise_percentages(group_name, len(group_expertise))
            
            for i, expertise_area in enumerate(group_expertise[:5]):  # Limit to 5 main areas
                # Find MEPs with relevant expertise
                specialized_meps = [
                    mep_name for mep_name in mep_list 
                    if mep_name in self.meps and 
                    any(exp.lower() in expertise_area.lower() for exp in self.meps[mep_name].expertise_areas)
                ]
                
                if specialized_meps and i < len(expertise_percentages):
                    # Select the first specialized MEP
                    specialized_mep_name = specialized_meps[0]
                    specialized_mep = self.meps[specialized_mep_name]
                    
                    if specialized_mep.agent:
                        # Assign percentage based on expertise importance
                        expertise_percentage = expertise_percentages[i]
                        
                        board_member = BoardMember(
                            agent=specialized_mep.agent,
                            role=BoardMemberRole.EXECUTIVE_DIRECTOR,
                            voting_weight=expertise_percentage,
                            expertise_areas=[expertise_area]
                        )
                        board_members.append(board_member)
                        board_member_percentages[specialized_mep_name] = expertise_percentage
            
            # Create the political group board with individual percentages
            political_group_board = PoliticalGroupBoard(
                group_name=group_name,
                members=mep_list,
                board_members=board_members,
                expertise_areas=group_expertise,
                voting_weight=voting_weight,
                group_speaker=group_speaker,
                total_meps=len(mep_list),
                board_member_percentages=board_member_percentages
            )
            
            # Create BoardOfDirectorsSwarm for this political group
            if board_members:
                agents = [member.agent for member in board_members if member.agent is not None]
                
                political_group_board.board_swarm = BoardOfDirectorsSwarm(
                    name=f"{group_name} Board",
                    description=f"Specialized board for {group_name} with expertise in {', '.join(group_expertise)}",
                    board_members=board_members,
                    agents=agents,
                    max_loops=3,
                    verbose=self.verbose,
                    decision_threshold=0.6,
                    enable_voting=True,
                    enable_consensus=True
                )
            
            self.political_group_boards[group_name] = political_group_board
            
            if self.verbose:
                logger.info(f"Created {group_name} board with {len(board_members)} members, "
                           f"voting weight: {voting_weight:.1%}, expertise: {', '.join(group_expertise[:3])}")
                logger.info(f"Board member percentages: {board_member_percentages}")

    def _assign_group_expertise(self, group_name: str, expertise_areas: Dict[str, List[str]]) -> List[str]:
        """Assign specialized expertise areas based on political group ideology."""
        
        # Map political groups to their primary expertise areas
        group_expertise_mapping = {
            "Group of the European People's Party (Christian Democrats)": [
                "economics", "law", "foreign", "social"
            ],
            "Group of the Progressive Alliance of Socialists and Democrats in the European Parliament": [
                "social", "economics", "environment", "law"
            ],
            "Renew Europe Group": [
                "economics", "technology", "environment", "foreign"
            ],
            "European Conservatives and Reformists Group": [
                "law", "foreign", "economics", "regional"
            ],
            "Group of the Greens/European Free Alliance": [
                "environment", "social", "technology", "agriculture"
            ],
            "The Left group in the European Parliament - GUE/NGL": [
                "social", "economics", "environment", "law"
            ],
            "Patriots for Europe Group": [
                "foreign", "law", "regional", "social"
            ],
            "Europe of Sovereign Nations Group": [
                "foreign", "law", "regional", "economics"
            ],
            "Non-attached Members": [
                "law", "foreign", "economics", "social"
            ]
        }
        
        # Get primary expertise areas for this group
        primary_areas = group_expertise_mapping.get(group_name, ["economics", "law", "social"])
        
        # Expand to specific expertise topics
        specific_expertise = []
        for area in primary_areas:
            if area in expertise_areas:
                specific_expertise.extend(expertise_areas[area])
        
        return specific_expertise[:8]  # Limit to 8 areas

    def _calculate_expertise_percentages(self, group_name: str, num_expertise_areas: int) -> List[float]:
        """Calculate individual percentages for board members based on political group and expertise areas."""
        
        # Define percentage distributions based on political group characteristics
        percentage_distributions = {
            "Group of the European People's Party (Christian Democrats)": [0.25, 0.20, 0.15, 0.05],  # CEO gets 35%
            "Group of the Progressive Alliance of Socialists and Democrats in the European Parliament": [0.25, 0.20, 0.15, 0.05],
            "Renew Europe Group": [0.30, 0.20, 0.10, 0.05],  # More emphasis on first expertise
            "Group of the Greens/European Free Alliance": [0.30, 0.20, 0.10, 0.05],
            "European Conservatives and Reformists Group": [0.25, 0.20, 0.15, 0.05],
            "The Left group in the European Parliament - GUE/NGL": [0.25, 0.20, 0.15, 0.05],
            "Patriots for Europe Group": [0.30, 0.20, 0.10, 0.05],
            "Europe of Sovereign Nations Group": [0.30, 0.20, 0.10, 0.05],
            "Non-attached Members": [0.40, 0.20, 0.05, 0.00]  # More concentrated power
        }
        
        # Get the distribution for this group
        distribution = percentage_distributions.get(group_name, [0.25, 0.20, 0.15, 0.05])
        
        # Return the appropriate number of percentages
        return distribution[:num_expertise_areas]

    def _create_parliament_speaker(self):
        """Create the Parliament Speaker who aggregates decisions from all political groups."""
        
        # Create parliament speaker agent
        speaker_agent = Agent(
            name="Parliament Speaker",
            system_prompt=self._generate_speaker_system_prompt(),
            llm="gpt-4",
            verbose=self.verbose
        )
        
        # Calculate majority threshold
        majority_threshold = (len(self.meps) // 2) + 1
        
        self.parliament_speaker = ParliamentSpeaker(
            name="Parliament Speaker",
            agent=speaker_agent,
            political_groups=self.political_group_boards,
            total_meps=len(self.meps),
            majority_threshold=majority_threshold
        )
        
        if self.verbose:
            logger.info(f"Created Parliament Speaker with majority threshold: {majority_threshold}")

    def _generate_speaker_system_prompt(self) -> str:
        """Generate system prompt for the Parliament Speaker."""
        
        return f"""You are the Parliament Speaker of the European Parliament, responsible for:
        
1. **Aggregating Political Group Decisions**: Collect and analyze decisions from all political groups
2. **Weighted Voting Calculation**: Calculate final results based on each group's percentage representation
3. **Majority Determination**: Determine if a proposal passes based on weighted majority
4. **Consensus Building**: Facilitate dialogue between groups when needed
5. **Transparent Reporting**: Provide clear explanations of voting results

**Political Group Distribution**:
{self._format_political_group_distribution()}

**Voting Rules**:
- Each political group votes as a unified board
- Group votes are weighted by their percentage of total MEPs
- Majority threshold: {self.parliament_speaker.majority_threshold if self.parliament_speaker else 'TBD'} MEPs
- Final decision: Positive, Negative, or Abstained

**Your Role**: Be impartial, transparent, and ensure democratic representation of all political groups.
"""

    def _format_political_group_distribution(self) -> str:
        """Format political group distribution for the speaker prompt."""
        
        if not self.political_group_boards:
            return "No political groups available"
        
        lines = []
        for group_name, board in self.political_group_boards.items():
            percentage = board.voting_weight * 100
            lines.append(f"- {group_name}: {board.total_meps} MEPs ({percentage:.1f}%)")
        
        return "\n".join(lines)
    
    def introduce_bill(
        self,
        title: str,
        description: str,
        bill_type: VoteType,
        committee: str,
        sponsor: str,
        co_sponsors: List[str] = None
    ) -> ParliamentaryBill:
        """
        Introduce a new bill to the parliament.
        
        Args:
            title: Bill title
            description: Bill description
            bill_type: Type of legislative procedure
            committee: Primary committee
            sponsor: Sponsoring MEP
            co_sponsors: List of co-sponsoring MEPs
            
        Returns:
            ParliamentaryBill: The introduced bill
        """
        if sponsor not in self.meps:
            raise ValueError(f"Sponsor {sponsor} is not a valid MEP")
        
        if committee not in self.committees:
            raise ValueError(f"Committee {committee} does not exist")
        
        bill = ParliamentaryBill(
            title=title,
            description=description,
            bill_type=bill_type,
            committee=committee,
            sponsor=sponsor,
            co_sponsors=co_sponsors or []
        )
        
        self.bills.append(bill)
        self.committees[committee].current_bills.append(bill)
        
        logger.info(f"Bill '{title}' introduced by {sponsor} in {committee} committee")
        return bill
    
    def conduct_committee_hearing(
        self,
        committee: str,
        bill: ParliamentaryBill,
        participants: List[str] = None
    ) -> Dict[str, Any]:
        """
        Conduct a committee hearing on a bill with cost optimization.
        
        Args:
            committee: Committee name
            bill: Bill under consideration
            participants: List of MEPs to participate
            
        Returns:
            Dict[str, Any]: Hearing results and transcript
        """
        if committee not in self.committees:
            raise ValueError(f"Committee {committee} does not exist")
        
        # Check budget before starting
        if not self.cost_tracker.check_budget():
            return {"error": "Budget exceeded", "cost_stats": self.cost_tracker.get_stats()}
        
        committee_meps = self.committees[committee].members
        if not participants:
            participants = committee_meps[:10]  # Limit to 10 participants
        
        # Check cache first
        cache_key = self._get_cache_key(f"committee_hearing_{committee}_{bill.title}", participants)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return {
                "committee": committee,
                "bill": bill.title,
                "participants": participants,
                "responses": cached_result,
                "date": datetime.now(),
                "cached": True,
                "cost_stats": self.cost_tracker.get_stats()
            }
        
        hearing_prompt = f"""
        Committee Hearing: {committee}
        Bill: {bill.title}
        Description: {bill.description}
        
        As a member of the {committee} committee, please provide your analysis and recommendations for this bill.
        Consider:
        1. Technical feasibility and legal compliance
        2. Impact on European citizens and businesses
        3. Alignment with EU policies and values
        4. Potential amendments or improvements
        5. Your recommendation for the full parliament
        
        Provide a detailed analysis with specific recommendations.
        """
        
        # Load MEP agents in batches
        all_responses = {}
        total_processed = 0
        
        for i in range(0, len(participants), self.batch_size):
            batch_participants = participants[i:i + self.batch_size]
            
            # Check budget for this batch
            if not self.cost_tracker.check_budget():
                logger.warning(f"Budget exceeded after processing {total_processed} participants")
                break
            
            # Load agents for this batch
            batch_agents = self._load_mep_agents_batch(batch_participants)
            
            if not batch_agents:
                continue
            
            # Run batch
            try:
                batch_results = run_agents_concurrently(batch_agents, hearing_prompt)
                
                # Map results back to participant names
                for j, agent in enumerate(batch_agents):
                    if j < len(batch_results):
                        participant_name = batch_participants[j]
                        all_responses[participant_name] = batch_results[j]
                        total_processed += 1
                
                # Estimate tokens used
                estimated_tokens = len(batch_agents) * 500  # ~500 tokens per response
                self.cost_tracker.add_tokens(estimated_tokens)
                
                if self.verbose:
                    logger.info(f"Processed committee hearing batch {i//self.batch_size + 1}: {len(batch_agents)} participants")
                
            except Exception as e:
                logger.error(f"Error processing committee hearing batch: {e}")
                continue
        
        # Cache the results
        if all_responses:
            self._cache_response(cache_key, str(all_responses))
        
        hearing_result = {
            "committee": committee,
            "bill": bill.title,
            "participants": participants[:total_processed],
            "responses": all_responses,
            "date": datetime.now(),
            "cached": False,
            "cost_stats": self.cost_tracker.get_stats(),
            "recommendations": self._synthesize_committee_recommendations(all_responses)
        }
        
        logger.info(f"Committee hearing completed for {bill.title} in {committee}")
        return hearing_result
    
    def _synthesize_committee_recommendations(self, responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Synthesize committee recommendations from individual responses.
        
        Args:
            responses: Dictionary of MEP responses
            
        Returns:
            Dict[str, Any]: Synthesized recommendations
        """
        # Simple synthesis - in a real implementation, this would be more sophisticated
        support_count = 0
        oppose_count = 0
        amend_count = 0
        
        for response in responses.values():
            response_lower = response.lower()
            if any(word in response_lower for word in ["support", "approve", "recommend", "favorable"]):
                support_count += 1
            elif any(word in response_lower for word in ["oppose", "reject", "against", "unfavorable"]):
                oppose_count += 1
            elif any(word in response_lower for word in ["amend", "modify", "improve", "revise"]):
                amend_count += 1
        
        total = len(responses)
        
        return {
            "support_percentage": (support_count / total) * 100 if total > 0 else 0,
            "oppose_percentage": (oppose_count / total) * 100 if total > 0 else 0,
            "amend_percentage": (amend_count / total) * 100 if total > 0 else 0,
            "recommendation": "support" if support_count > oppose_count else "oppose" if oppose_count > support_count else "amend"
        }
    
    def conduct_parliamentary_debate(
        self,
        bill: ParliamentaryBill,
        participants: List[str] = None,
        max_speakers: int = 20
    ) -> Dict[str, Any]:
        """
        Conduct a parliamentary debate on a bill with cost optimization.
        
        Args:
            bill: Bill under debate
            participants: List of MEPs to participate
            max_speakers: Maximum number of speakers
            
        Returns:
            Dict[str, Any]: Debate transcript and analysis
        """
        # Check budget before starting
        if not self.cost_tracker.check_budget():
            return {"error": "Budget exceeded", "cost_stats": self.cost_tracker.get_stats()}
        
        if not participants:
            # Select diverse participants from different political groups
            participants = []
            for group_name, meps in self.political_groups.items():
                if len(meps) > 0:
                    participants.extend(meps[:3])  # 3 MEPs per group
                    if len(participants) >= max_speakers:
                        break
        
        participants = participants[:max_speakers]
        
        # Check cache first
        cache_key = self._get_cache_key(f"parliamentary_debate_{bill.title}", participants)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return {
                "bill": bill.title,
                "participants": participants,
                "transcript": cached_result,
                "date": datetime.now(),
                "cached": True,
                "cost_stats": self.cost_tracker.get_stats()
            }
        
        debate_prompt = f"""
        Parliamentary Debate: {bill.title}
        
        You are participating in a parliamentary debate on this bill. Please provide your position and arguments.
        
        Bill Description: {bill.description}
        Bill Type: {bill.bill_type.value}
        
        Consider:
        1. Your political group's position on this issue
        2. Impact on your country and constituents
        3. European-wide implications
        4. Your areas of expertise
        5. Potential amendments or alternatives
        
        Provide a clear, reasoned argument for your position.
        """
        
        # Conduct debate with batching
        debate_transcript = []
        total_processed = 0
        
        for i in range(0, len(participants), self.batch_size):
            batch_participants = participants[i:i + self.batch_size]
            
            # Check budget for this batch
            if not self.cost_tracker.check_budget():
                logger.warning(f"Budget exceeded after processing {total_processed} speakers")
                break
            
            # Load agents for this batch
            batch_agents = self._load_mep_agents_batch(batch_participants)
            
            if not batch_agents:
                continue
            
            # Run batch
            try:
                batch_results = run_agents_concurrently(batch_agents, debate_prompt)
                
                # Create debate entries
                for j, agent in enumerate(batch_agents):
                    if j < len(batch_results):
                        participant_name = batch_participants[j]
                        mep = self.meps[participant_name]
                        
                        debate_entry = {
                            "speaker": participant_name,
                            "political_group": mep.political_group,
                            "country": mep.country,
                            "position": batch_results[j],
                            "timestamp": datetime.now()
                        }
                        debate_transcript.append(debate_entry)
                        total_processed += 1
                
                # Estimate tokens used
                estimated_tokens = len(batch_agents) * 500  # ~500 tokens per response
                self.cost_tracker.add_tokens(estimated_tokens)
                
                if self.verbose:
                    logger.info(f"Processed debate batch {i//self.batch_size + 1}: {len(batch_agents)} speakers")
                
            except Exception as e:
                logger.error(f"Error processing debate batch: {e}")
                continue
        
        # Cache the results
        if debate_transcript:
            self._cache_response(cache_key, str(debate_transcript))
        
        debate_result = {
            "bill": bill.title,
            "participants": participants[:total_processed],
            "transcript": debate_transcript,
            "date": datetime.now(),
            "cached": False,
            "cost_stats": self.cost_tracker.get_stats(),
            "analysis": self._analyze_debate(debate_transcript)
        }
        
        self.debates.append(debate_result)
        logger.info(f"Parliamentary debate completed for {bill.title} with {total_processed} speakers")
        return debate_result
    
    def _analyze_debate(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze debate transcript for key themes and positions.
        
        Args:
            transcript: Debate transcript
            
        Returns:
            Dict[str, Any]: Debate analysis
        """
        # Simple analysis - in a real implementation, this would use NLP
        support_count = 0
        oppose_count = 0
        neutral_count = 0
        
        for entry in transcript:
            position = entry["position"].lower()
            if any(word in position for word in ["support", "approve", "favorable", "yes"]):
                support_count += 1
            elif any(word in position for word in ["oppose", "reject", "against", "no"]):
                oppose_count += 1
            else:
                neutral_count += 1
        
        total = len(transcript)
        
        return {
            "support_count": support_count,
            "oppose_count": oppose_count,
            "neutral_count": neutral_count,
            "support_percentage": (support_count / total) * 100 if total > 0 else 0,
            "oppose_percentage": (oppose_count / total) * 100 if total > 0 else 0,
            "neutral_percentage": (neutral_count / total) * 100 if total > 0 else 0
        }
    
    def conduct_democratic_vote(
        self,
        bill: ParliamentaryBill,
        participants: List[str] = None
    ) -> ParliamentaryVote:
        """
        Conduct a democratic vote on a bill using the Board of Directors pattern with lazy loading.
        
        Args:
            bill: Bill to vote on
            participants: List of MEPs to participate
            
        Returns:
            ParliamentaryVote: Vote results
        """
        # Check budget before starting
        if not self.cost_tracker.check_budget():
            return ParliamentaryVote(
                bill=bill,
                vote_type=bill.bill_type,
                result=VoteResult.FAILED
            )
        
        if not participants:
            participants = list(self.meps.keys())
        
        # Use democratic swarm for decision-making if available
        democratic_result = None
        if self.democratic_swarm is not None:
            decision_task = f"""
            Parliamentary Vote: {bill.title}
            
            Bill Description: {bill.description}
            Bill Type: {bill.bill_type.value}
            
            As a democratic decision-making body, please:
            1. Analyze the bill's merits and implications
            2. Consider the interests of all European citizens
            3. Evaluate alignment with European values and policies
            4. Make a democratic decision on whether to support or oppose this bill
            5. Provide reasoning for your decision
            
            This is a critical legislative decision that will affect all EU citizens.
            """
            
            # Get democratic decision
            democratic_result = self.democratic_swarm.run_board_meeting(decision_task)
        
        # Conduct individual MEP votes with lazy loading
        individual_votes = {}
        reasoning = {}
        total_processed = 0
        
        # Process participants in batches
        for i in range(0, len(participants), self.batch_size):
            batch_participants = participants[i:i + self.batch_size]
            
            # Check budget for this batch
            if not self.cost_tracker.check_budget():
                logger.warning(f"Budget exceeded after processing {total_processed} voters")
                break
            
            # Load agents for this batch
            batch_agents = self._load_mep_agents_batch(batch_participants)
            
            if not batch_agents:
                continue
            
            # Create voting prompt
            vote_prompt = f"""
            Vote on Bill: {bill.title}
            
            {bill.description}
            
            {f"Democratic Council Decision: {democratic_result.plan}" if democratic_result else "No democratic council decision available."}
            
            As an MEP, please vote on this bill. Consider:
            1. The democratic council's analysis (if available)
            2. Your political group's position
            3. Your constituents' interests
            4. European-wide implications
            
            Respond with 'FOR', 'AGAINST', or 'ABSTAIN' and explain your reasoning.
            """
            
            # Run batch voting
            try:
                batch_results = run_agents_concurrently(batch_agents, vote_prompt)
                
                # Process results
                for j, agent in enumerate(batch_agents):
                    if j < len(batch_results):
                        participant_name = batch_participants[j]
                        response = batch_results[j]
                        
                        # Parse vote
                        response_lower = response.lower()
                        if any(word in response_lower for word in ["for", "support", "yes", "approve"]):
                            vote = "FOR"
                        elif any(word in response_lower for word in ["against", "oppose", "no", "reject"]):
                            vote = "AGAINST"
                        else:
                            vote = "ABSTAIN"
                        
                        individual_votes[participant_name] = vote
                        reasoning[participant_name] = response
                        total_processed += 1
                
                # Estimate tokens used
                estimated_tokens = len(batch_agents) * 500  # ~500 tokens per response
                self.cost_tracker.add_tokens(estimated_tokens)
                
                if self.verbose:
                    logger.info(f"Processed voting batch {i//self.batch_size + 1}: {len(batch_agents)} voters")
                
            except Exception as e:
                logger.error(f"Error processing voting batch: {e}")
                continue
        
        # Calculate results
        votes_for = sum(1 for vote in individual_votes.values() if vote == "FOR")
        votes_against = sum(1 for vote in individual_votes.values() if vote == "AGAINST")
        abstentions = sum(1 for vote in individual_votes.values() if vote == "ABSTAIN")
        absent = len(participants) - len(individual_votes)
        
        # Determine result
        if votes_for > votes_against:
            result = VoteResult.PASSED
        elif votes_against > votes_for:
            result = VoteResult.FAILED
        else:
            result = VoteResult.TIED
        
        vote_result = ParliamentaryVote(
            bill=bill,
            vote_type=bill.bill_type,
            votes_for=votes_for,
            votes_against=votes_against,
            abstentions=abstentions,
            absent=absent,
            result=result,
            individual_votes=individual_votes,
            reasoning=reasoning
        )
        
        self.votes.append(vote_result)
        bill.status = "voted"
        
        logger.info(f"Democratic vote completed for {bill.title}: {result.value} ({total_processed} voters processed)")
        return vote_result
    
    def conduct_hierarchical_democratic_vote(
        self,
        bill: ParliamentaryBill,
        participants: List[str] = None
    ) -> ParliamentaryVote:
        """
        Conduct a hierarchical democratic vote using political group boards and parliament speaker.
        
        This enhanced voting system:
        1. Each political group votes internally as a specialized board
        2. Group speakers (CEOs) synthesize their group's position
        3. Parliament Speaker aggregates all group decisions based on percentage representation
        4. Final result calculated using weighted voting
        
        Args:
            bill: Bill to vote on
            participants: List of MEPs to participate (optional, uses all by default)
            
        Returns:
            ParliamentaryVote: Enhanced vote results with group-level analysis
        """
        
        if not self.enable_hierarchical_democracy:
            logger.warning("Hierarchical democracy not enabled, falling back to standard voting")
            return self.conduct_democratic_vote(bill, participants)
        
        logger.info(f"Conducting hierarchical democratic vote on: {bill.title}")
        
        # Initialize vote tracking
        vote = ParliamentaryVote(
            bill=bill,
            vote_type=bill.bill_type,
            date=datetime.now()
        )
        
        # Step 1: Each political group votes internally
        group_decisions = {}
        group_reasoning = {}
        
        for group_name, group_board in self.political_group_boards.items():
            if not group_board.board_swarm:
                continue
                
            logger.info(f"Conducting internal vote for {group_name}")
            
            # Create voting task for this group
            voting_task = f"""
            Parliamentary Vote: {bill.title}
            
            Bill Description: {bill.description}
            Bill Type: {bill.bill_type.value}
            Committee: {bill.committee}
            Sponsor: {bill.sponsor}
            
            As a specialized board representing {group_name} with expertise in {', '.join(group_board.expertise_areas[:3])}, 
            please analyze this bill and provide your group's position.
            
            Consider:
            1. How does this bill align with your political group's values and priorities?
            2. What are the economic, social, and legal implications?
            3. How does it affect your areas of expertise?
            4. What amendments or modifications would you suggest?
            
            Provide your group's decision: POSITIVE, NEGATIVE, or ABSTAIN
            Include detailed reasoning for your position.
            """
            
            try:
                # Get group decision using their specialized board
                group_result = group_board.board_swarm.run(voting_task)
                
                # Parse the group decision
                group_decision = self._parse_group_decision(group_result)
                group_decisions[group_name] = group_decision
                group_reasoning[group_name] = group_result
                
                logger.info(f"{group_name} decision: {group_decision}")
                
            except Exception as e:
                logger.error(f"Error in {group_name} vote: {e}")
                group_decisions[group_name] = "ABSTAIN"
                group_reasoning[group_name] = f"Error during voting: {str(e)}"
        
        # Step 2: Parliament Speaker aggregates group decisions
        if self.parliament_speaker and self.parliament_speaker.agent:
            logger.info("Parliament Speaker aggregating group decisions")
            
            aggregation_task = f"""
            Parliamentary Vote Aggregation: {bill.title}
            
            Political Group Decisions:
            {self._format_group_decisions(group_decisions, group_reasoning)}
            
            Political Group Distribution:
            {self._format_political_group_distribution()}
            
            As Parliament Speaker, calculate the final result based on:
            1. Each group's decision (POSITIVE/NEGATIVE/ABSTAIN)
            2. Each group's voting weight (percentage of parliament)
            3. Majority threshold: {self.parliament_speaker.majority_threshold} MEPs
            
            Provide:
            1. Final result: PASSED, FAILED, or TIED
            2. Vote counts: For, Against, Abstentions
            3. Weighted analysis of each group's contribution
            4. Summary of the democratic process
            """
            
            try:
                speaker_result = self.parliament_speaker.agent.run(aggregation_task)
                
                # Parse speaker's analysis
                final_result = self._parse_speaker_analysis(speaker_result, group_decisions)
                
                # Update vote with results
                vote.result = final_result['result']
                vote.votes_for = final_result['votes_for']
                vote.votes_against = final_result['votes_against']
                vote.abstentions = final_result['abstentions']
                vote.individual_votes = group_decisions
                vote.reasoning = group_reasoning
                
                logger.info(f"Final result: {vote.result.value}")
                logger.info(f"Votes - For: {vote.votes_for}, Against: {vote.votes_against}, Abstain: {vote.abstentions}")
                
            except Exception as e:
                logger.error(f"Error in speaker aggregation: {e}")
                # Fallback to simple counting
                vote = self._fallback_vote_calculation(vote, group_decisions)
        
        # Store the vote
        self.votes.append(vote)
        
        return vote

    def _parse_group_decision(self, group_result: str) -> str:
        """Parse the decision from a political group's voting result."""
        
        result_lower = group_result.lower()
        
        if any(word in result_lower for word in ['positive', 'for', 'support', 'approve', 'pass']):
            return "POSITIVE"
        elif any(word in result_lower for word in ['negative', 'against', 'oppose', 'reject', 'fail']):
            return "NEGATIVE"
        else:
            return "ABSTAIN"

    def _format_group_decisions(self, group_decisions: Dict[str, str], group_reasoning: Dict[str, str]) -> str:
        """Format group decisions for the speaker's analysis."""
        
        lines = []
        for group_name, decision in group_decisions.items():
            board = self.political_group_boards.get(group_name)
            if board:
                percentage = board.voting_weight * 100
                reasoning = group_reasoning.get(group_name, "No reasoning provided")
                lines.append(f"- {group_name} ({board.total_meps} MEPs, {percentage:.1f}%): {decision}")
                lines.append(f"  Reasoning: {reasoning[:200]}...")
        
        return "\n".join(lines)

    def _parse_speaker_analysis(self, speaker_result: str, group_decisions: Dict[str, str]) -> Dict[str, Any]:
        """Parse the Parliament Speaker's analysis to extract final vote results using dual-layer percentage system."""
        
        # Initialize counters
        votes_for = 0
        votes_against = 0
        abstentions = 0
        
        # Calculate weighted votes using dual-layer percentage system
        for group_name, decision in group_decisions.items():
            board = self.political_group_boards.get(group_name)
            if board and board.board_member_percentages:
                # Calculate weighted votes using individual board member percentages
                group_weighted_votes = self._calculate_group_weighted_votes(board, decision)
                
                if decision == "POSITIVE":
                    votes_for += group_weighted_votes
                elif decision == "NEGATIVE":
                    votes_against += group_weighted_votes
                else:  # ABSTAIN
                    abstentions += group_weighted_votes
            else:
                # Fallback to simple calculation if no individual percentages available
                if board:
                    weighted_votes = int(board.total_meps * board.voting_weight)
                    
                    if decision == "POSITIVE":
                        votes_for += weighted_votes
                    elif decision == "NEGATIVE":
                        votes_against += weighted_votes
                    else:  # ABSTAIN
                        abstentions += weighted_votes
        
        # Determine result
        if votes_for > votes_against:
            result = VoteResult.PASSED
        elif votes_against > votes_for:
            result = VoteResult.FAILED
        else:
            result = VoteResult.TIED
        
        return {
            'result': result,
            'votes_for': votes_for,
            'votes_against': votes_against,
            'abstentions': abstentions
        }

    def _calculate_group_weighted_votes(self, board: PoliticalGroupBoard, decision: str) -> int:
        """Calculate weighted votes for a political group using individual board member percentages."""
        
        total_weighted_votes = 0
        
        # Calculate votes based on individual board member percentages
        for member_name, internal_percentage in board.board_member_percentages.items():
            # Convert internal percentage to parliament percentage
            # internal_percentage is percentage within the group
            # board.voting_weight is group's percentage of parliament
            parliament_percentage = internal_percentage * board.voting_weight
            
            # Calculate weighted votes for this member
            member_weighted_votes = int(board.total_meps * parliament_percentage)
            total_weighted_votes += member_weighted_votes
            
            if self.verbose:
                logger.debug(f"{member_name}: {internal_percentage:.1%} of {board.group_name} "
                           f"({board.voting_weight:.1%} of parliament) = {parliament_percentage:.3%} "
                           f"= {member_weighted_votes} weighted votes")
        
        return total_weighted_votes

    def _fallback_vote_calculation(self, vote: ParliamentaryVote, group_decisions: Dict[str, str]) -> ParliamentaryVote:
        """Fallback vote calculation if speaker analysis fails."""
        
        votes_for = 0
        votes_against = 0
        abstentions = 0
        
        for group_name, decision in group_decisions.items():
            board = self.political_group_boards.get(group_name)
            if board:
                if decision == "POSITIVE":
                    votes_for += board.total_meps
                elif decision == "NEGATIVE":
                    votes_against += board.total_meps
                else:
                    abstentions += board.total_meps
        
        vote.votes_for = votes_for
        vote.votes_against = votes_against
        vote.abstentions = abstentions
        
        if votes_for > votes_against:
            vote.result = VoteResult.PASSED
        elif votes_against > votes_for:
            vote.result = VoteResult.FAILED
        else:
            vote.result = VoteResult.TIED
        
        return vote
    
    def get_parliament_composition(self) -> Dict[str, Any]:
        """
        Get the current composition of the parliament including cost statistics.
        
        Returns:
            Dict[str, Any]: Parliament composition statistics
        """
        composition = {
            "total_meps": len(self.meps),
            "loaded_meps": len([mep for mep in self.meps.values() if mep.is_loaded]),
            "political_groups": {},
            "countries": {},
            "leadership": {},
            "committees": {},
            "cost_stats": self.cost_tracker.get_stats(),
            "optimization": {
                "lazy_loading": self.enable_lazy_loading,
                "caching": self.enable_caching,
                "batch_size": self.batch_size,
                "budget_limit": self.cost_tracker.budget_limit
            }
        }
        
        # Political group breakdown
        for group_name, meps in self.political_groups.items():
            composition["political_groups"][group_name] = {
                "count": len(meps),
                "percentage": (len(meps) / len(self.meps)) * 100
            }
        
        # Country breakdown
        country_counts = {}
        for mep in self.meps.values():
            country = mep.country
            country_counts[country] = country_counts.get(country, 0) + 1
        
        composition["countries"] = country_counts
        
        # Leadership positions
        leadership = {}
        for mep in self.meps.values():
            if mep.role != ParliamentaryRole.MEP:
                role = mep.role.value
                if role not in leadership:
                    leadership[role] = []
                leadership[role].append(mep.full_name)
        
        composition["leadership"] = leadership
        
        # Committee composition
        for committee_name, committee in self.committees.items():
            composition["committees"][committee_name] = {
                "chair": committee.chair,
                "vice_chair": committee.vice_chair,
                "member_count": len(committee.members),
                "current_bills": len(committee.current_bills)
            }
        
        return composition
    
    def get_cost_statistics(self) -> Dict[str, Any]:
        """
        Get detailed cost statistics for the parliamentary operations.
        
        Returns:
            Dict[str, Any]: Cost statistics and optimization metrics
        """
        stats = self.cost_tracker.get_stats()
        
        # Add additional metrics
        stats.update({
            "total_meps": len(self.meps),
            "loaded_meps": len([mep for mep in self.meps.values() if mep.is_loaded]),
            "loading_efficiency": len([mep for mep in self.meps.values() if mep.is_loaded]) / len(self.meps) if self.meps else 0,
            "cache_size": len(self.response_cache),
            "optimization_enabled": {
                "lazy_loading": self.enable_lazy_loading,
                "caching": self.enable_caching,
                "batching": self.batch_size > 1
            }
        })
        
        return stats
    
    def run_optimized_parliamentary_session(
        self,
        bill_title: str,
        bill_description: str,
        bill_type: VoteType = VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
        committee: str = "Legal Affairs",
        sponsor: str = None,
        max_cost: float = 50.0
    ) -> Dict[str, Any]:
        """
        Run a complete parliamentary session with cost optimization.
        
        Args:
            bill_title: Title of the bill
            bill_description: Description of the bill
            bill_type: Type of legislative procedure
            committee: Primary committee
            sponsor: Sponsoring MEP (random if not specified)
            max_cost: Maximum cost for this session
            
        Returns:
            Dict[str, Any]: Complete session results with cost tracking
        """
        # Set temporary budget for this session
        original_budget = self.cost_tracker.budget_limit
        self.cost_tracker.budget_limit = min(original_budget, max_cost)
        
        try:
            # Select sponsor if not provided
            if not sponsor:
                sponsor = random.choice(list(self.meps.keys()))
            
            # Introduce bill
            bill = self.introduce_bill(
                title=bill_title,
                description=bill_description,
                bill_type=bill_type,
                committee=committee,
                sponsor=sponsor
            )
            
            # Conduct committee hearing
            hearing = self.conduct_committee_hearing(committee, bill)
            
            # Conduct parliamentary debate
            debate = self.conduct_parliamentary_debate(bill)
            
            # Conduct democratic vote
            vote = self.conduct_democratic_vote(bill)
            
            session_result = {
                "bill": bill,
                "hearing": hearing,
                "debate": debate,
                "vote": vote,
                "cost_stats": self.cost_tracker.get_stats(),
                "session_summary": {
                    "bill_title": bill_title,
                    "sponsor": sponsor,
                    "committee": committee,
                    "hearing_recommendation": hearing.get("recommendations", {}).get("recommendation", "unknown"),
                    "debate_support_percentage": debate.get("analysis", {}).get("support_percentage", 0),
                    "vote_result": vote.result.value,
                    "final_outcome": "PASSED" if vote.result == VoteResult.PASSED else "FAILED",
                    "total_cost": self.cost_tracker.total_cost_estimate
                }
            }
            
            logger.info(f"Optimized parliamentary session completed for {bill_title}: {session_result['session_summary']['final_outcome']}")
            logger.info(f"Session cost: ${self.cost_tracker.total_cost_estimate:.2f}")
            
            return session_result
            
        finally:
            # Restore original budget
            self.cost_tracker.budget_limit = original_budget
    
    def run_democratic_session(
        self,
        bill_title: str,
        bill_description: str,
        bill_type: VoteType = VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
        committee: str = "Legal Affairs",
        sponsor: str = None
    ) -> Dict[str, Any]:
        """
        Run a complete democratic parliamentary session on a bill.
        
        Args:
            bill_title: Title of the bill
            bill_description: Description of the bill
            bill_type: Type of legislative procedure
            committee: Primary committee
            sponsor: Sponsoring MEP (random if not specified)
            
        Returns:
            Dict[str, Any]: Complete session results
        """
        # Select sponsor if not provided
        if not sponsor:
            sponsor = random.choice(list(self.meps.keys()))
        
        # Introduce bill
        bill = self.introduce_bill(
            title=bill_title,
            description=bill_description,
            bill_type=bill_type,
            committee=committee,
            sponsor=sponsor
        )
        
        # Conduct committee hearing
        hearing = self.conduct_committee_hearing(committee, bill)
        
        # Conduct parliamentary debate
        debate = self.conduct_parliamentary_debate(bill)
        
        # Conduct democratic vote
        vote = self.conduct_democratic_vote(bill)
        
        session_result = {
            "bill": bill,
            "hearing": hearing,
            "debate": debate,
            "vote": vote,
            "session_summary": {
                "bill_title": bill_title,
                "sponsor": sponsor,
                "committee": committee,
                "hearing_recommendation": hearing["recommendations"]["recommendation"],
                "debate_support_percentage": debate["analysis"]["support_percentage"],
                "vote_result": vote.result.value,
                "final_outcome": "PASSED" if vote.result == VoteResult.PASSED else "FAILED"
            }
        }
        
        logger.info(f"Democratic session completed for {bill_title}: {session_result['session_summary']['final_outcome']}")
        return session_result
    
    def run_hierarchical_democratic_session(
        self,
        bill_title: str,
        bill_description: str,
        bill_type: VoteType = VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
        committee: str = "Legal Affairs",
        sponsor: str = None
    ) -> Dict[str, Any]:
        """
        Run a complete hierarchical democratic session from bill introduction to final vote.
        
        This enhanced session uses:
        1. Political group boards with specialized expertise
        2. Group-level internal voting and discussion
        3. Parliament Speaker aggregation of group decisions
        4. Weighted voting based on political group percentages
        
        Args:
            bill_title: Title of the bill
            bill_description: Description of the bill
            bill_type: Type of legislative procedure
            committee: Committee responsible for the bill
            sponsor: MEP sponsoring the bill
            
        Returns:
            Dict[str, Any]: Complete session results including group decisions and final vote
        """
        
        if not self.enable_hierarchical_democracy:
            logger.warning("Hierarchical democracy not enabled, falling back to standard session")
            return self.run_democratic_session(bill_title, bill_description, bill_type, committee, sponsor)
        
        logger.info(f"Starting hierarchical democratic session: {bill_title}")
        
        # Step 1: Introduce the bill
        if not sponsor:
            sponsor = list(self.meps.keys())[0]  # Use first MEP as sponsor
        
        bill = self.introduce_bill(
            title=bill_title,
            description=bill_description,
            bill_type=bill_type,
            committee=committee,
            sponsor=sponsor
        )
        
        # Step 2: Conduct committee hearing (if enabled)
        committee_result = None
        if self.enable_committee_work:
            logger.info(f"Conducting committee hearing in {committee}")
            committee_result = self.conduct_committee_hearing(committee, bill)
        
        # Step 3: Conduct parliamentary debate (if enabled)
        debate_result = None
        if self.enable_democratic_discussion:
            logger.info("Conducting parliamentary debate")
            debate_result = self.conduct_parliamentary_debate(bill)
        
        # Step 4: Conduct hierarchical democratic vote
        logger.info("Conducting hierarchical democratic vote")
        vote_result = self.conduct_hierarchical_democratic_vote(bill)
        
        # Step 5: Compile comprehensive session report
        session_report = {
            "session_type": "hierarchical_democratic",
            "bill": {
                "title": bill.title,
                "description": bill.description,
                "type": bill.bill_type.value,
                "committee": bill.committee,
                "sponsor": bill.sponsor,
                "status": bill.status
            },
            "committee_work": committee_result,
            "parliamentary_debate": debate_result,
            "vote_results": {
                "final_result": vote_result.result.value,
                "votes_for": vote_result.votes_for,
                "votes_against": vote_result.votes_against,
                "abstentions": vote_result.abstentions,
                "total_votes": vote_result.votes_for + vote_result.votes_against + vote_result.abstentions
            },
            "political_group_decisions": vote_result.individual_votes,
            "group_reasoning": vote_result.reasoning,
            "parliament_composition": self.get_parliament_composition(),
            "session_summary": self._generate_hierarchical_session_summary(bill, vote_result)
        }
        
        logger.info(f"Hierarchical democratic session completed. Final result: {vote_result.result.value}")
        
        return session_report

    def _generate_hierarchical_session_summary(self, bill: ParliamentaryBill, vote: ParliamentaryVote) -> str:
        """Generate a summary of the hierarchical democratic session with dual-layer percentage breakdown."""
        
        total_votes = vote.votes_for + vote.votes_against + vote.abstentions
        participation_rate = (total_votes / len(self.meps)) * 100 if self.meps else 0
        
        summary = f"""
🏛️ HIERARCHICAL DEMOCRATIC SESSION SUMMARY

📋 Bill: {bill.title}
📊 Final Result: {vote.result.value}
📈 Participation Rate: {participation_rate:.1f}%

🗳️ VOTE BREAKDOWN:
• For: {vote.votes_for} votes
• Against: {vote.votes_against} votes  
• Abstentions: {vote.abstentions} votes

🏛️ POLITICAL GROUP DECISIONS (Dual-Layer Percentage System):
"""
        
        for group_name, decision in vote.individual_votes.items():
            board = self.political_group_boards.get(group_name)
            if board:
                group_percentage = board.voting_weight * 100
                summary += f"\n• {group_name}: {decision} ({board.total_meps} MEPs, {group_percentage:.1f}% of parliament)"
                
                # Show individual board member percentages
                if board.board_member_percentages:
                    summary += f"\n  📊 Board Member Breakdown:"
                    for member_name, internal_percentage in board.board_member_percentages.items():
                        parliament_percentage = internal_percentage * board.voting_weight * 100
                        summary += f"\n    - {member_name}: {internal_percentage:.1%} of group = {parliament_percentage:.3f}% of parliament"
        
        summary += f"\n\n🎯 DUAL-LAYER DEMOCRATIC PROCESS:"
        summary += f"\n• Each political group operates as a specialized board"
        summary += f"\n• Board members have individual percentages within their group"
        summary += f"\n• Individual percentages × Group percentage = Parliament percentage"
        summary += f"\n• Parliament Speaker aggregates all weighted decisions"
        summary += f"\n• Final result based on {len(self.political_group_boards)} political groups with {sum(len(board.board_member_percentages) for board in self.political_group_boards.values())} board members"
        
        return summary
    
    def get_mep(self, mep_name: str) -> Optional[ParliamentaryMember]:
        """
        Get a specific MEP by name.
        
        Args:
            mep_name: Name of the MEP
            
        Returns:
            Optional[ParliamentaryMember]: MEP if found, None otherwise
        """
        return self.meps.get(mep_name)
    
    def get_committee(self, committee_name: str) -> Optional[ParliamentaryCommittee]:
        """
        Get a specific committee by name.
        
        Args:
            committee_name: Name of the committee
            
        Returns:
            Optional[ParliamentaryCommittee]: Committee if found, None otherwise
        """
        return self.committees.get(committee_name)
    
    def get_political_group_members(self, group_name: str) -> List[str]:
        """
        Get all MEPs in a specific political group.
        
        Args:
            group_name: Name of the political group
            
        Returns:
            List[str]: List of MEP names in the group
        """
        return self.political_groups.get(group_name, [])
    
    def get_country_members(self, country: str) -> List[str]:
        """
        Get all MEPs from a specific country.
        
        Args:
            country: Name of the country
            
        Returns:
            List[str]: List of MEP names from the country
        """
        return [mep_name for mep_name, mep in self.meps.items() if mep.country == country] 

    def _load_wikipedia_personalities(self):
        """Load Wikipedia personality profiles for MEPs."""
        
        if not self.enable_wikipedia_personalities:
            return
        
        try:
            # Initialize personality scraper
            self.personality_scraper = WikipediaPersonalityScraper(
                output_dir="mep_personalities",
                verbose=self.verbose
            )
            
            # Load existing personality profiles
            personality_dir = "mep_personalities"
            if os.path.exists(personality_dir):
                profile_files = [f for f in os.listdir(personality_dir) if f.endswith('.json')]
                
                for filename in profile_files:
                    filepath = os.path.join(personality_dir, filename)
                    try:
                        profile = self.personality_scraper.load_personality_profile(filepath)
                        self.personality_profiles[profile.full_name] = profile
                        
                        if self.verbose:
                            logger.debug(f"Loaded personality profile: {profile.full_name}")
                            
                    except Exception as e:
                        logger.warning(f"Error loading personality profile {filename}: {e}")
                
                if self.verbose:
                    logger.info(f"Loaded {len(self.personality_profiles)} Wikipedia personality profiles")
            else:
                if self.verbose:
                    logger.info("No existing personality profiles found. Run Wikipedia scraper to create profiles.")
                    
        except Exception as e:
            logger.error(f"Error loading Wikipedia personalities: {e}")
            self.enable_wikipedia_personalities = False

    def scrape_wikipedia_personalities(self, delay: float = 1.0) -> Dict[str, str]:
        """
        Scrape Wikipedia personality data for all MEPs.
        
        Args:
            delay: Delay between requests to be respectful to Wikipedia
            
        Returns:
            Dictionary mapping MEP names to their personality profile file paths
        """
        
        if not self.enable_wikipedia_personalities:
            logger.error("Wikipedia personality system not available")
            return {}
        
        if not self.personality_scraper:
            self.personality_scraper = WikipediaPersonalityScraper(
                output_dir="mep_personalities",
                verbose=self.verbose
            )
        
        logger.info("Starting Wikipedia personality scraping for all MEPs...")
        profile_files = self.personality_scraper.scrape_all_mep_personalities(
            xml_file=self.eu_data_file,
            delay=delay
        )
        
        # Reload personality profiles
        self._load_wikipedia_personalities()
        
        return profile_files

    def get_mep_personality_profile(self, mep_name: str) -> Optional[MEPPersonalityProfile]:
        """
        Get personality profile for a specific MEP.
        
        Args:
            mep_name: Name of the MEP
            
        Returns:
            MEPPersonalityProfile if found, None otherwise
        """
        return self.personality_profiles.get(mep_name)

    def analyze_political_landscape(self, bill: ParliamentaryBill) -> Dict[str, Any]:
        """
        Analyze the political landscape for a bill to predict voting outcomes.
        
        Args:
            bill: Bill to analyze
            
        Returns:
            Dict[str, Any]: Political analysis results
        """
        analysis = {
            "overall_support": 0.0,
            "opposition": 0.0,
            "uncertainty": 0.0,
            "group_analysis": {}
        }
        
        # Analyze by political group
        for group_name, meps in self.political_groups.items():
            if not meps:
                continue
            
            # Simple analysis based on political group alignment
            group_support = 0.0
            group_opposition = 0.0
            
            # Assign support based on political group characteristics
            if "Green" in group_name or "Environment" in bill.description:
                group_support = 75.0
                group_opposition = 15.0
            elif "Socialist" in group_name or "Social" in bill.description:
                group_support = 70.0
                group_opposition = 20.0
            elif "Conservative" in group_name or "Economic" in bill.description:
                group_support = 60.0
                group_opposition = 30.0
            elif "Liberal" in group_name or "Digital" in bill.description:
                group_support = 65.0
                group_opposition = 25.0
            else:
                group_support = 50.0
                group_opposition = 30.0
            
            group_uncertainty = 100.0 - group_support - group_opposition
            
            analysis["group_analysis"][group_name] = {
                "support": group_support,
                "opposition": group_opposition,
                "uncertainty": group_uncertainty,
                "mep_count": len(meps)
            }
        
        # Calculate overall support weighted by group size
        total_meps = len(self.meps)
        if total_meps > 0:
            weighted_support = 0.0
            weighted_opposition = 0.0
            weighted_uncertainty = 0.0
            
            for group_name, group_data in analysis["group_analysis"].items():
                weight = group_data["mep_count"] / total_meps
                weighted_support += group_data["support"] * weight
                weighted_opposition += group_data["opposition"] * weight
                weighted_uncertainty += group_data["uncertainty"] * weight
            
            analysis["overall_support"] = weighted_support
            analysis["opposition"] = weighted_opposition
            analysis["uncertainty"] = weighted_uncertainty
        
        return analysis
