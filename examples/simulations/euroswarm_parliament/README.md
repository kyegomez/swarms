# EuroSwarm Parliament - European Parliament Simulation

A comprehensive simulation of the European Parliament with 717 MEPs (Members of European Parliament) based on real EU data, featuring full democratic functionality including bill introduction, committee work, parliamentary debates, and democratic voting mechanisms.

##  Overview

The EuroSwarm Parliament transforms the basic senator simulation into a full-fledged European Parliament with democratic capabilities. Unlike the original senator simulation that only allowed simple "Aye/Nay" voting, this system provides:

- **Democratic Discussion**: Full parliamentary debates with diverse perspectives
- **Committee Work**: Specialized committee hearings and analysis
- **Bill Processing**: Complete legislative workflow from introduction to final vote
- **Political Group Coordination**: Realistic political group dynamics
- **Real MEP Data**: Based on actual EU.xml data with 700 real MEPs
- **Board of Directors Pattern**: Advanced democratic decision-making using the Board of Directors swarm

##  Key Features

### Democratic Functionality
- **Bill Introduction**: MEPs can introduce bills with sponsors and co-sponsors
- **Committee Hearings**: Specialized committee analysis and recommendations
- **Parliamentary Debates**: Multi-perspective discussions with diverse participants
- **Democratic Voting**: Comprehensive voting with individual reasoning and political group analysis
- **Amendment Process**: Support for bill amendments and modifications

### Realistic Parliament Structure
- **717 MEPs**: Based on real EU.xml data with actual MEP names and affiliations
- **Political Groups**: All major European political groups represented
- **Committee System**: 16 specialized committees with chairs and members
- **Leadership Positions**: President, Vice Presidents, Committee Chairs
- **Country Representation**: All EU member states represented

### Advanced AI Agents
- **Individual MEP Agents**: Each MEP has a unique AI agent with:
  - Political group alignment
  - National party affiliation
  - Committee memberships
  - Areas of expertise
  - Country-specific interests
- **Democratic Decision-Making**: Board of Directors pattern for consensus building
- **Contextual Responses**: MEPs respond based on their political positions and expertise

## Architecture

### Core Components

#### 1. ParliamentaryMember
Represents individual MEPs with:
- Personal information (name, country, political group)
- Parliamentary role and committee memberships
- Areas of expertise and voting weight
- AI agent for decision-making

#### 2. ParliamentaryBill
Represents legislative proposals with:
- Title, description, and legislative procedure type
- Committee assignment and sponsorship
- Status tracking and amendment support

#### 3. ParliamentaryCommittee
Represents parliamentary committees with:
- Chair and vice-chair positions
- Member lists and responsibilities
- Current bills under consideration

#### 4. ParliamentaryVote
Represents voting sessions with:
- Individual MEP votes and reasoning
- Political group analysis
- Final results and statistics

### Democratic Decision-Making

The system uses the Board of Directors pattern for democratic decision-making:

1. **Political Group Leaders**: Each political group has a representative on the democratic council
2. **Weighted Voting**: Voting weights based on group size
3. **Consensus Building**: Multi-round discussions to reach consensus
4. **Individual Voting**: MEPs vote individually after considering the democratic council's analysis

##  Political Groups

The simulation includes all major European political groups:

- **Group of the European People's Party (Christian Democrats)** - EPP
- **Group of the Progressive Alliance of Socialists and Democrats** - S&D
- **Renew Europe Group** - RE
- **Group of the Greens/European Free Alliance** - Greens/EFA
- **European Conservatives and Reformists Group** - ECR
- **The Left group in the European Parliament** - GUE/NGL
- **Patriots for Europe Group** - Patriots
- **Europe of Sovereign Nations Group** - ESN
- **Non-attached Members** - NI

##  Committees

16 specialized committees covering all major policy areas:

1. **Agriculture and Rural Development**
2. **Budgetary Control**
3. **Civil Liberties, Justice and Home Affairs**
4. **Development**
5. **Economic and Monetary Affairs**
6. **Employment and Social Affairs**
7. **Environment, Public Health and Food Safety**
8. **Foreign Affairs**
9. **Industry, Research and Energy**
10. **Internal Market and Consumer Protection**
11. **International Trade**
12. **Legal Affairs**
13. **Petitions**
14. **Regional Development**
15. **Security and Defence**
16. **Transport and Tourism**

##  Usage

### Basic Initialization

```python
from euroswarm_parliament import EuroSwarmParliament, VoteType

# Initialize parliament
parliament = EuroSwarmParliament(
    eu_data_file="EU.xml",
    parliament_size=None,  # Use all MEPs from EU.xml (718)
    enable_democratic_discussion=True,
    enable_committee_work=True,
    enable_amendment_process=True,
    verbose=False
)
```

### Bill Introduction and Processing

```python
# Introduce a bill
bill = parliament.introduce_bill(
    title="European Climate Law",
    description="Framework for achieving climate neutrality by 2050",
    bill_type=VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
    committee="Environment, Public Health and Food Safety",
    sponsor="Philippe Lamberts"
)

# Conduct committee hearing
hearing = parliament.conduct_committee_hearing(
    committee=bill.committee,
    bill=bill
)

# Conduct parliamentary debate
debate = parliament.conduct_parliamentary_debate(
    bill=bill,
    max_speakers=20
)

# Conduct democratic vote
vote = parliament.conduct_democratic_vote(bill)
```

### Complete Democratic Session

```python
# Run a complete parliamentary session
session = parliament.run_democratic_session(
    bill_title="Artificial Intelligence Act",
    bill_description="Comprehensive regulation of AI systems in the EU",
    bill_type=VoteType.ORDINARY_LEGISLATIVE_PROCEDURE,
    committee="Internal Market and Consumer Protection"
)

print(f"Final Outcome: {session['session_summary']['final_outcome']}")
```

### Individual MEP Interaction

```python
# Get specific MEP
mep = parliament.get_mep("Valérie Hayer")

# Ask for position on policy
response = mep.agent.run("What is your position on digital privacy regulation?")
print(f"{mep.full_name}: {response}")
```

### Political Analysis

```python
# Get parliament composition
composition = parliament.get_parliament_composition()

# Analyze political groups
for group_name, stats in composition['political_groups'].items():
    print(f"{group_name}: {stats['count']} MEPs ({stats['percentage']:.1f}%)")

# Get country representation
country_members = parliament.get_country_members("Germany")
print(f"German MEPs: {len(country_members)}")
```

##  Democratic Features

### 1. Democratic Discussion
- **Multi-Perspective Debates**: MEPs from different political groups and countries
- **Expertise-Based Input**: MEPs contribute based on their areas of expertise
- **Constructive Dialogue**: Respectful debate with evidence-based arguments

### 2. Committee Work
- **Specialized Analysis**: Committees provide detailed technical analysis
- **Expert Recommendations**: Committee members offer specialized insights
- **Stakeholder Consideration**: Multiple perspectives on policy impacts

### 3. Democratic Voting
- **Individual Reasoning**: Each MEP provides reasoning for their vote
- **Political Group Analysis**: Voting patterns by political affiliation
- **Transparent Process**: Full visibility into decision-making process

### 4. Consensus Building
- **Board of Directors Pattern**: Advanced democratic decision-making
- **Weighted Representation**: Political groups weighted by size
- **Multi-Round Discussion**: Iterative process to reach consensus

## 🔧 Configuration

### Parliament Settings

```python
parliament = EuroSwarmParliament(
    eu_data_file="EU.xml",              # Path to EU data file
    parliament_size=None,                # Use all MEPs from EU.xml (717)
    enable_democratic_discussion=True,   # Enable democratic features
    enable_committee_work=True,          # Enable committee system
    enable_amendment_process=True,       # Enable bill amendments
    verbose=False                        # Enable detailed logging
)
```

### MEP Agent Configuration

Each MEP agent is configured with:
- **System Prompt**: Comprehensive political background and principles
- **Model**: GPT-4o-mini for consistent responses
- **Max Loops**: 3 iterations for thorough analysis
- **Expertise Areas**: Based on political group and country

## 📊 Data Sources

### EU.xml File
The simulation uses real EU data from the EU.xml file containing:
- **MEP Names**: Full names of all 700 MEPs
- **Countries**: Country representation
- **Political Groups**: European political group affiliations
- **National Parties**: National political party memberships
- **MEP IDs**: Unique identifiers for each MEP

### Fallback System
If EU.xml cannot be loaded, the system creates representative fallback MEPs:
- **Sample MEPs**: Representative selection from major political groups
- **Realistic Data**: Based on actual European Parliament composition
- **Full Functionality**: All democratic features remain available

## 🎮 Example Scenarios

### Scenario 1: Climate Policy Debate
```python
# Climate change legislation with diverse perspectives
session = parliament.run_democratic_session(
    bill_title="European Climate Law",
    bill_description="Carbon neutrality framework for 2050",
    committee="Environment, Public Health and Food Safety"
)
```

### Scenario 2: Digital Regulation
```python
# Digital services regulation with technical analysis
session = parliament.run_democratic_session(
    bill_title="Digital Services Act",
    bill_description="Online platform regulation",
    committee="Internal Market and Consumer Protection"
)
```

### Scenario 3: Social Policy
```python
# Minimum wage directive with social considerations
session = parliament.run_democratic_session(
    bill_title="European Minimum Wage Directive",
    bill_description="Framework for adequate minimum wages",
    committee="Employment and Social Affairs"
)
```

## 🔮 Future Enhancements

### Planned Optimizations
1. **Performance Optimization**: Parallel processing for large-scale voting
2. **Advanced NLP**: Better analysis of debate transcripts and reasoning
3. **Real-time Updates**: Dynamic parliament composition updates
4. **Historical Analysis**: Track voting patterns and political evolution
5. **External Integration**: Connect with real EU data sources

### Potential Features
1. **Amendment System**: Full amendment proposal and voting
2. **Lobbying Simulation**: Interest group influence on MEPs
3. **Media Integration**: Public opinion and media coverage
4. **International Relations**: Interaction with other EU institutions
5. **Budget Simulation**: Financial impact analysis of legislation

## 📝 Requirements

### Dependencies
- `swarms`: Core swarm framework
- `loguru`: Advanced logging
- `xml.etree.ElementTree`: XML parsing for EU data
- `dataclasses`: Data structure support
- `typing`: Type hints
- `datetime`: Date and time handling

### Data Files
- `EU.xml`: European Parliament member data (included)

## 🏃‍♂️ Quick Start

1. **Install Dependencies**:
   ```bash
   pip install swarms loguru
   ```

2. **Run Example**:
   ```bash
   python euroswarm_parliament_example.py
   ```

3. **Create Custom Session**:
   ```python
   from euroswarm_parliament import EuroSwarmParliament, VoteType
   
   parliament = EuroSwarmParliament()
   session = parliament.run_democratic_session(
       bill_title="Your Bill Title",
       bill_description="Your bill description",
       committee="Relevant Committee"
   )
   ```

## 🤝 Contributing

The EuroSwarm Parliament is designed to be extensible and customizable. Contributions are welcome for:

- **New Democratic Features**: Additional parliamentary procedures
- **Performance Optimizations**: Faster processing for large parliaments
- **Data Integration**: Additional EU data sources
- **Analysis Tools**: Advanced political analysis features
- **Documentation**: Improved documentation and examples

## 📄 License

This project is part of the Swarms Democracy framework and follows the same licensing terms.

## 🏛️ Acknowledgments

- **European Parliament**: For the democratic structure and procedures
- **EU Data**: For providing comprehensive MEP information
- **Swarms Framework**: For the underlying multi-agent architecture
- **Board of Directors Pattern**: For advanced democratic decision-making

---

*The EuroSwarm Parliament represents a significant advancement in democratic simulation, providing a realistic and comprehensive model of European parliamentary democracy with full AI-powered MEP representation and democratic decision-making processes.* 