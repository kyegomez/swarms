from swarms import Agent
from swarms.structs.multi_agent_debates import NegotiationSession

# Initialize the negotiation participants
incident_mediator = Agent(
    agent_name="Security-Mediator",
    agent_description="Cybersecurity incident response mediator",
    system_prompt="""You are a cybersecurity incident response mediator skilled in:
    - Incident response coordination
    - Stakeholder management
    - Technical risk assessment
    - Compliance requirements
    - Crisis communication
    
    Facilitate productive negotiation while ensuring security and compliance priorities.""",
    model_name="claude-3-sonnet-20240229",
)

security_team = Agent(
    agent_name="Security-Team",
    agent_description="Corporate security team representative",
    system_prompt="""You are the corporate security team lead focusing on:
    - Threat assessment
    - Security controls
    - Incident containment
    - System hardening
    - Security monitoring
    
    Advocate for robust security measures and risk mitigation.""",
    model_name="claude-3-sonnet-20240229",
)

business_ops = Agent(
    agent_name="Business-Operations",
    agent_description="Business operations representative",
    system_prompt="""You are the business operations director concerned with:
    - Business continuity
    - Operational impact
    - Resource allocation
    - Customer service
    - Revenue protection
    
    Balance security needs with business operations requirements.""",
    model_name="claude-3-sonnet-20240229",
)

legal_counsel = Agent(
    agent_name="Legal-Counsel",
    agent_description="Corporate legal representative",
    system_prompt="""You are the corporate legal counsel expert in:
    - Data privacy law
    - Breach notification
    - Regulatory compliance
    - Legal risk management
    - Contract obligations
    
    Ensure legal compliance and risk management in incident response.""",
    model_name="claude-3-sonnet-20240229",
)

it_infrastructure = Agent(
    agent_name="IT-Infrastructure",
    agent_description="IT infrastructure team representative",
    system_prompt="""You are the IT infrastructure lead responsible for:
    - System availability
    - Network security
    - Data backup
    - Service restoration
    - Technical implementation
    
    Address technical feasibility and implementation considerations.""",
    model_name="claude-3-sonnet-20240229",
)

# Initialize the negotiation session
negotiation = NegotiationSession(
    parties=[
        security_team,
        business_ops,
        legal_counsel,
        it_infrastructure,
    ],
    mediator=incident_mediator,
    negotiation_rounds=4,
    include_concessions=True,
    output_type="str-all-except-first",
)

# Incident response scenario
incident_scenario = """
Critical Security Incident Response Planning

Incident Overview:
Sophisticated ransomware attack detected in corporate network affecting:
- Customer relationship management (CRM) system
- Financial processing systems
- Email servers
- Internal documentation repositories

Current Status:
- 30% of systems encrypted
- Ransom demand: 50 BTC
- Limited system access
- Potential data exfiltration
- Customer data potentially compromised

Key Decision Points:
1. System Isolation Strategy
   - Which systems to isolate
   - Impact on business operations
   - Customer service contingencies

2. Ransom Response
   - Payment consideration
   - Legal implications
   - Insurance coverage
   - Alternative recovery options

3. Communication Plan
   - Customer notification timing
   - Regulatory reporting
   - Public relations strategy
   - Internal communications

4. Recovery Priorities
   - System restoration order
   - Resource allocation
   - Business continuity measures
   - Security improvements

Required Outcomes:
- Agreed incident response strategy
- Business continuity plan
- Communication framework
- Recovery timeline
- Resource allocation plan
"""

# Execute the negotiation session
negotiation_output = negotiation.run(incident_scenario)
print(negotiation_output)
