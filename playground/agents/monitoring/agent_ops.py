"""
* WORKING

What this script does:
Multi-Agent run to test AgentOps (https://www.agentops.ai/)

Requirements:
1. Create an account on https://www.agentops.ai/ and run pip install agentops
2. Add the folowing API key(s) in your .env file:
   - OPENAI_API_KEY
   - AGENTOPS_API_KEY
3. Go to your agentops dashboard to observe your activity

"""

################ Adding project root to PYTHONPATH ################################
# If you are running playground examples in the project files directly, use this: 

import sys
import os

sys.path.insert(0, os.getcwd())

################ Adding project root to PYTHONPATH ################################

from swarms import Agent, OpenAIChat, AgentRearrange

Treasurer = Agent(
    agent_name="Treasurer",
    system_prompt="Give your opinion on the cash management.",
    agent_description=(
        "responsible for managing an organization's financial assets and liquidity. They oversee cash management, " 
        "investment strategies, and financial risk. Key duties include monitoring cash flow, managing bank relationships, " 
        "ensuring sufficient funds for operations, and optimizing returns on short-term investments. Treasurers also often " 
        "handle debt management and may be involved in capital raising activities."
    ),
    llm=OpenAIChat(),
    max_loops=1,
    agent_ops_on=True,
)


CFO = Agent(
    agent_name="CFO",
    system_prompt="Give your opinion on the financial performance of the company.",
    agent_description=(
    "the top financial executive in an organization, overseeing all financial operations and strategy. Their role is broader than a treasurer's and includes:\n"
    "Financial planning and analysis\n"
    "Accounting and financial reporting\n"
    "Budgeting and forecasting\n"
    "Strategic financial decision-making\n"
    "Compliance and risk management\n"
    "Investor relations (in public companies)\n"
    "Overseeing the finance and accounting departments"
    ),
    llm=OpenAIChat(),
    max_loops=1,
    agent_ops_on=True,
)

swarm = AgentRearrange(
    agents=[Treasurer, CFO],
    flow="Treasurer -> CFO",
)

results = swarm.run("Date,Revenue,Expenses,Profit,Cash_Flow,Inventory,Customer_Acquisition_Cost,Customer_Retention_Rate,Marketing_Spend,R&D_Spend,Debt,Assets\n"
        "2023-01-01,1000000,800000,200000,150000,500000,100,0.85,50000,100000,2000000,5000000\n"
        "2023-02-01,1050000,820000,230000,180000,520000,95,0.87,55000,110000,1950000,5100000\n"
        "2023-03-01,1100000,850000,250000,200000,530000,90,0.88,60000,120000,1900000,5200000\n"
        "2023-04-01,1200000,900000,300000,250000,550000,85,0.90,70000,130000,1850000,5400000\n"
        "2023-05-01,1300000,950000,350000,300000,580000,80,0.92,80000,140000,1800000,5600000\n"
        "2023-06-01,1400000,1000000,400000,350000,600000,75,0.93,90000,150000,1750000,5800000\n"
        "2023-07-01,1450000,1050000,400000,320000,620000,78,0.91,95000,160000,1700000,5900000\n"
        "2023-08-01,1500000,1100000,400000,300000,650000,80,0.90,100000,170000,1650000,6000000\n"
        "2023-09-01,1550000,1150000,400000,280000,680000,82,0.89,105000,180000,1600000,6100000\n"
        "2023-10-01,1600000,1200000,400000,260000,700000,85,0.88,110000,190000,1550000,6200000\n"
        "2023-11-01,1650000,1250000,400000,240000,720000,88,0.87,115000,200000,1500000,6300000\n"
        "2023-12-01,1700000,1300000,400000,220000,750000,90,0.86,120000,210000,1450000,6400000\n"
        "2024-01-01,1500000,1200000,300000,180000,780000,95,0.84,100000,180000,1500000,6300000\n"
        "2024-02-01,1550000,1220000,330000,200000,760000,92,0.85,105000,185000,1480000,6350000\n"
        "2024-03-01,1600000,1240000,360000,220000,740000,89,0.86,110000,190000,1460000,6400000\n"
        "2024-04-01,1650000,1260000,390000,240000,720000,86,0.87,115000,195000,1440000,6450000\n"
        "2024-05-01,1700000,1280000,420000,260000,700000,83,0.88,120000,200000,1420000,6500000\n"
        "2024-06-01,1750000,1300000,450000,280000,680000,80,0.89,125000,205000,1400000,6550000"
        )