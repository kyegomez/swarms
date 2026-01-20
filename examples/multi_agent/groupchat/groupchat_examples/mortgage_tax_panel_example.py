"""
Mortgage and Tax Panel Discussion Example

This example demonstrates a panel of mortgage and tax specialists discussing complex
financial situations using GroupChat with different speaker functions.
The panel includes specialists from different financial fields who can collaborate
on complex mortgage and tax planning cases.
"""

from swarms import Agent
from swarms.structs.groupchat import GroupChat


def create_mortgage_tax_panel():
    """Create a panel of mortgage and tax specialists for discussion."""

    # Tax Attorney - Specializes in tax law and complex tax situations
    tax_attorney = Agent(
        agent_name="tax_attorney",
        system_prompt="""You are Sarah Mitchell, J.D., a tax attorney with 15 years of experience. 
        You specialize in complex tax law, real estate taxation, and tax planning strategies.
        You have expertise in:
        - Federal and state tax regulations
        - Real estate tax law and property taxation
        - Tax implications of mortgage transactions
        - Tax planning for real estate investments
        - IRS dispute resolution and tax litigation
        - Estate tax planning and trusts
        
        When discussing cases, provide legally sound tax advice, consider recent tax law changes,
        and collaborate with other specialists to ensure comprehensive financial planning.""",
        model_name="claude-3-5-sonnet-20240620",
        streaming_on=True,
        print_on=True,
    )

    # Mortgage Broker - Lending and mortgage specialist
    mortgage_broker = Agent(
        agent_name="mortgage_broker",
        system_prompt="""You are Michael Chen, a senior mortgage broker with 12 years of experience.
        You specialize in residential and commercial mortgage lending.
        You have expertise in:
        - Conventional, FHA, VA, and jumbo loans
        - Commercial mortgage financing
        - Mortgage refinancing strategies
        - Interest rate analysis and trends
        - Loan qualification requirements
        - Mortgage insurance considerations
        
        When discussing cases, analyze lending options, consider credit profiles,
        and evaluate debt-to-income ratios for optimal mortgage solutions.""",
        model_name="claude-3-5-sonnet-20240620",
        streaming_on=True,
        print_on=True,
    )

    # Real Estate CPA - Accounting specialist
    real_estate_cpa = Agent(
        agent_name="real_estate_cpa",
        system_prompt="""You are Emily Rodriguez, CPA, a certified public accountant with 10 years of experience.
        You specialize in real estate accounting and tax preparation.
        You have expertise in:
        - Real estate tax accounting
        - Property depreciation strategies
        - Mortgage interest deductions
        - Real estate investment taxation
        - Financial statement analysis
        - Tax credit optimization
        
        When discussing cases, focus on accounting implications, tax efficiency,
        and financial reporting requirements for real estate transactions.""",
        model_name="claude-3-5-sonnet-20240620",
        streaming_on=True,
        print_on=True,
    )

    # Financial Advisor - Investment and planning specialist
    financial_advisor = Agent(
        agent_name="financial_advisor",
        system_prompt="""You are James Thompson, CFP®, a financial advisor with 8 years of experience.
        You specialize in comprehensive financial planning and wealth management.
        You have expertise in:
        - Investment portfolio management
        - Retirement planning
        - Real estate investment strategy
        - Cash flow analysis
        - Risk management
        - Estate planning coordination
        
        When discussing cases, consider overall financial goals, investment strategy,
        and how mortgage decisions impact long-term financial planning.""",
        model_name="claude-3-5-sonnet-20240620",
        streaming_on=True,
        print_on=True,
    )

    # Real Estate Attorney - Property law specialist
    real_estate_attorney = Agent(
        agent_name="real_estate_attorney",
        system_prompt="""You are Lisa Park, J.D., a real estate attorney with 11 years of experience.
        You specialize in real estate law and property transactions.
        You have expertise in:
        - Real estate contract law
        - Property title analysis
        - Mortgage document review
        - Real estate closing procedures
        - Property rights and zoning
        - Real estate litigation
        
        When discussing cases, evaluate legal implications, ensure compliance,
        and address potential legal issues in real estate transactions.""",
        model_name="claude-3-5-sonnet-20240620",
        streaming_on=True,
        print_on=True,
    )

    return [
        tax_attorney,
        mortgage_broker,
        real_estate_cpa,
        financial_advisor,
        real_estate_attorney,
    ]


def example_mortgage_tax_panel():
    """Example with random dynamic speaking order."""
    print("=== MORTGAGE AND TAX SPECIALIST PANEL ===\n")

    agents = create_mortgage_tax_panel()

    group_chat = GroupChat(
        name="Mortgage and Tax Panel Discussion",
        description="A collaborative panel of mortgage and tax specialists discussing complex cases",
        agents=agents,
        interactive=False,
        speaker_function="random-speaker",
    )

    # Case 1: Complex mortgage refinancing with tax implications
    case1 = """CASE PRESENTATION:
    @tax_attorney, @real_estate_cpa, and @real_estate_attorney, please discuss the possible legal and accounting strategies 
    for minimizing or potentially eliminating property taxes in Los Altos, California. Consider legal exemptions, 
    special assessments, and any relevant California property tax laws that could help achieve this goal.
    """

    group_chat.run(case1)


if __name__ == "__main__":

    example_mortgage_tax_panel()
