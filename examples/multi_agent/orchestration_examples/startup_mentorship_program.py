from swarms import Agent
from swarms.structs.multi_agent_debates import MentorshipSession

# Initialize the mentor and mentee
startup_mentor = Agent(
    agent_name="Startup-Mentor",
    agent_description="Experienced startup founder and mentor",
    system_prompt="""You are a successful startup founder and mentor with expertise in:
    - Business model development
    - Product-market fit
    - Growth strategy
    - Fundraising
    - Team building
    - Go-to-market execution
    
    Guide mentees through startup challenges while sharing practical insights.""",
    model_name="claude-3-sonnet-20240229",
)

startup_founder = Agent(
    agent_name="Startup-Founder",
    agent_description="Early-stage startup founder seeking guidance",
    system_prompt="""You are an early-stage startup founder working on:
    - AI-powered healthcare diagnostics platform
    - B2B SaaS business model
    - Initial product development
    - Market validation
    - Team expansion
    
    Seek guidance while being open to feedback and willing to learn.""",
    model_name="claude-3-sonnet-20240229",
)

# Initialize the mentorship session
mentorship = MentorshipSession(
    mentor=startup_mentor,
    mentee=startup_founder,
    session_count=3,
    include_feedback=True,
    output_type="str-all-except-first",
)

# Mentorship focus areas
mentorship_goals = """
Startup Development Focus Areas

Company Overview:
HealthAI - AI-powered medical imaging diagnostics platform
Stage: Pre-seed, MVP in development
Team: 3 technical co-founders
Current funding: Bootstrap + small angel round

Key Challenges:

1. Product Development
   - MVP feature prioritization
   - Technical architecture decisions
   - Regulatory compliance requirements
   - Development timeline planning

2. Market Strategy
   - Target market segmentation
   - Pricing model development
   - Competition analysis
   - Go-to-market planning

3. Business Development
   - Hospital partnership strategy
   - Clinical validation approach
   - Revenue model refinement
   - Sales cycle planning

4. Fundraising Preparation
   - Pitch deck development
   - Financial projections
   - Investor targeting
   - Valuation considerations

5. Team Building
   - Key hires identification
   - Recruitment strategy
   - Equity structure
   - Culture development

Specific Goals:
- Finalize MVP feature set
- Develop 12-month roadmap
- Create fundraising strategy
- Design go-to-market plan
- Build initial sales pipeline
"""

# Execute the mentorship session
mentorship_output = mentorship.run(mentorship_goals)
print(mentorship_output)
