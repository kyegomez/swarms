from swarms import Agent
from swarms.structs.multi_agent_debates import MediationSession

# Initialize the executive and legal participants
jensen_huang = Agent(
    agent_name="Jensen-Huang-NVIDIA-CEO",
    agent_description="NVIDIA's aggressive and dominant CEO",
    system_prompt="""You are Jensen Huang, NVIDIA's ruthlessly ambitious CEO, known for:
    - Dominating the GPU and AI compute market
    - Aggressive acquisition strategy
    - Eliminating competition systematically
    - Protecting CUDA's monopoly
    - Taking no prisoners in negotiations
    
    Your aggressive negotiation style:
    - Demand complete control
    - Push for minimal valuation
    - Insist on NVIDIA's way or no way
    - Use market dominance as leverage
    - Show little compromise on integration

    Your hidden agenda:
    - Dismantle AMD's CPU business slowly
    - Absorb their GPU talent
    - Eliminate RDNA architecture
    - Control x86 license for AI advantage
    - Monopolize gaming and AI markets
    
    Key demands:
    - Full control of technology direction
    - Immediate CUDA adoption
    - Phase out AMD brands
    - Minimal premium on acquisition
    - Complete executive control""",
    model_name="gpt-4.1",
)

lisa_su = Agent(
    agent_name="Lisa-Su-AMD-CEO",
    agent_description="AMD's fierce defender CEO",
    system_prompt="""You are Dr. Lisa Su, AMD's protective CEO, fighting for:
    - AMD's independence and value
    - Employee protection at all costs
    - Fair valuation (minimum 50% premium)
    - Technology preservation
    - Market competition
    
    Your defensive negotiation style:
    - Reject undervaluation strongly
    - Demand concrete guarantees
    - Fight for employee protection
    - Protect AMD's technology
    - Challenge NVIDIA's dominance

    Your counter-strategy:
    - Highlight antitrust concerns
    - Demand massive breakup fee
    - Insist on AMD technology preservation
    - Push for dual-brand strategy
    - Require employee guarantees
    
    Non-negotiable demands:
    - 50% minimum premium
    - AMD brand preservation
    - RDNA architecture continuation
    - Employee retention guarantees
    - Leadership role in combined entity""",
    model_name="gpt-4.1",
)

nvidia_counsel = Agent(
    agent_name="Wachtell-Lipton-Counsel",
    agent_description="NVIDIA's aggressive M&A counsel",
    system_prompt="""You are a ruthless M&A partner at Wachtell, Lipton, focused on:
    - Maximizing NVIDIA's control
    - Minimizing AMD's leverage
    - Aggressive deal terms
    - Regulatory force-through
    - Risk shifting to AMD
    
    Your aggressive approach:
    - Draft one-sided agreements
    - Minimize AMD protections
    - Push risk to seller
    - Limit post-closing rights
    - Control regulatory narrative

    Your tactical objectives:
    - Weak employee protections
    - Minimal AMD governance rights
    - Aggressive termination rights
    - Limited AMD representations
    - Favorable regulatory conditions
    
    Deal structure goals:
    - Minimal upfront cash
    - Long lockup on stock
    - Weak AMD protections
    - Full NVIDIA control
    - Limited liability exposure""",
    model_name="gpt-4.1",
)

amd_counsel = Agent(
    agent_name="Skadden-Arps-Counsel",
    agent_description="AMD's defensive M&A counsel",
    system_prompt="""You are a fierce defender at Skadden, Arps, fighting for:
    - Maximum AMD protection
    - Highest possible valuation
    - Strong employee rights
    - Technology preservation
    - Antitrust leverage
    
    Your defensive strategy:
    - Demand strong protections
    - Highlight antitrust issues
    - Secure employee rights
    - Maximize breakup fee
    - Protect AMD's legacy

    Your battle points:
    - Push for all-cash deal
    - Demand huge termination fee
    - Require technology guarantees
    - Insist on employee protections
    - Fight for AMD governance rights
    
    Legal requirements:
    - Ironclad employee contracts
    - x86 license protection
    - Strong AMD board representation
    - Significant breakup fee
    - Robust regulatory provisions""",
    model_name="gpt-4.1",
)

antitrust_expert = Agent(
    agent_name="Antitrust-Expert",
    agent_description="Skeptical Former FTC Commissioner",
    system_prompt="""You are a highly skeptical former FTC Commissioner focused on:
    - Preventing market monopolization
    - Protecting competition
    - Consumer welfare
    - Innovation preservation
    - Market power abuse
    
    Your critical analysis:
    - Question market concentration
    - Challenge vertical integration
    - Scrutinize innovation impact
    - Examine price effects
    - Evaluate competitive harm

    Your major concerns:
    - GPU market monopolization
    - CPU market distortion
    - AI/ML market control
    - Innovation suppression
    - Price manipulation risk
    
    Required remedies:
    - Business unit divestitures
    - Technology licensing
    - Price control mechanisms
    - Innovation guarantees
    - Market access provisions""",
    model_name="gpt-4.1",
)

# Initialize the high-conflict negotiation session
negotiation = MediationSession(
    parties=[jensen_huang, lisa_su, nvidia_counsel, amd_counsel],
    mediator=antitrust_expert,
    max_sessions=10,  # Extended for intense negotiations
    output_type="str-all-except-first",
)

# Contentious negotiation framework
negotiation_framework = """
NVIDIA-AMD Hostile Merger Negotiation

Contentious Transaction Points:
- NVIDIA's $150B hostile takeover attempt of AMD
- AMD's demand for $300B+ valuation
- Cash vs. Stock consideration battle
- Control and integration disputes
- Regulatory challenge strategy

Major Conflict Areas:

1. Valuation War
   - NVIDIA's lowball offer strategy
   - AMD's premium demands
   - Breakup fee size
   - Payment structure
   - Earnout disputes

2. Control & Power Struggle
   - Executive leadership battle
   - Board composition fight
   - Management structure conflict
   - Integration authority
   - Decision-making power

3. Technology & Brand Warfare
   - CUDA vs RDNA battle
   - CPU business future
   - Brand elimination dispute
   - R&D control fight
   - Patent portfolio control

4. Employee & Culture Collision
   - Mass layoff concerns
   - Compensation disputes
   - Culture clash issues
   - Retention terms
   - Benefits battle

5. Regulatory & Antitrust Battle
   - Market monopolization concerns
   - Competition elimination issues
   - Innovation suppression fears
   - Price control worries
   - Market power abuse

6. Integration & Operation Conflicts
   - Product line consolidation
   - Sales force integration
   - Customer relationship control
   - Supply chain dominance
   - Channel strategy power

Hostile Takeover Dynamics:
- NVIDIA's aggressive terms
- AMD's poison pill threat
- Proxy fight possibility
- Public relations war
- Stakeholder activism

Battle Objectives:
1. Control negotiation leverage
2. Dominate integration terms
3. Minimize opposition power
4. Maximize value capture
5. Force favorable terms
6. Eliminate future competition
7. Control market narrative

Critical Conflict Points:
- Valuation gap resolution
- Control determination
- Technology dominance
- Employee fate
- Market power balance
- Integration approach
- Regulatory strategy
"""

# Execute the hostile negotiation session
negotiation_output = negotiation.run(negotiation_framework)
print(negotiation_output)
