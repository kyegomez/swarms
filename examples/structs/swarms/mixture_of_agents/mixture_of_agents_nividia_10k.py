import os
from swarms import Agent
from swarm_models import OpenAIChat
from swarms.structs.mixture_of_agents import MixtureOfAgents


SEC_DATA = """
Where You Can Find More Information
Investors and others should note that we announce material financial information to our investors using our investor relations website, press releases, SEC filings and public conference calls and webcasts. We also use the following social media channels as a means of disclosing information about the company, our products, our planned financial and other announcements and attendance at upcoming investor and industry conferences, and other matters, and for complying with our disclosure obligations under Regulation FD: 
NVIDIA Corporate Blog (http://blogs.nvidia.com)
NVIDIA Technical Blog (http://developer.nvidia.com/blog/)
NVIDIA LinkedIn Page (http://www.linkedin.com/company/nvidia)
NVIDIA Facebook Page (https://www.facebook.com/nvidia)
NVIDIA Instagram Page (https://www.instagram.com/nvidia)
NVIDIA X Account (https://x.com/nvidia)
In addition, investors and others can view NVIDIA videos on YouTube (https://www.YouTube.com/nvidia).
The information we post through these social media channels may be deemed material. Accordingly, investors should monitor these accounts and the blog, in addition to following our press releases, SEC filings and public conference calls and webcasts. This list may be updated from time to time. The information we post through these channels is not a part of this Quarterly Report on Form 10-Q. These channels may be updated from time to time on NVIDIA's investor relations website.
2

Part I. Financial Information
Item 1. Financial Statements (Unaudited)

NVIDIA Corporation and Subsidiaries
Condensed Consolidated Statements of Income
(In millions, except per share data)
(Unaudited)

 	Three Months Ended
 	Apr 28, 2024		Apr 30, 2023
Revenue	$	26,044 			$	7,192 	
Cost of revenue	5,638 			2,544 	
Gross profit	20,406 			4,648 	
Operating expenses	 		 
Research and development	2,720 			1,875 	
Sales, general and administrative	777 			633 	
Total operating expenses	3,497 			2,508 	
Operating income	16,909 			2,140 	
Interest income	359 			150 	
Interest expense	(64)			(66)	
Other, net	75 			(15)	
Other income (expense), net
370 			69 	
Income before income tax	17,279 			2,209 	
Income tax expense	2,398 			166 	
Net income	$	14,881 			$	2,043 	
Net income per share:			
Basic	$	6.04 			$	0.83 	
Diluted	$	5.98 			$	0.82 	
Weighted average shares used in per share computation:			
Basic	2,462 			2,470 	
Diluted	2,489 			2,490 	
 

See accompanying Notes to Condensed Consolidated Financial Statements.
3

NVIDIA Corporation and Subsidiaries
Condensed Consolidated Statements of Comprehensive Income
(In millions)
(Unaudited)
 	Three Months Ended
 	Apr 28, 2024		Apr 30, 2023
 			
Net income	$	14,881 			$	2,043 	
Other comprehensive loss, net of tax			
Available-for-sale securities:			
Net change in unrealized gain (loss)	(128)			17 	
Cash flow hedges:			
Net change in unrealized loss	(4)			(13)	
Reclassification adjustments for net realized loss included in net income	(4)			(11)	
Net change in unrealized loss	(8)			(24)	
Other comprehensive loss, net of tax	(136)			(7)	
Total comprehensive income	$	14,745 			$	2,036 	
 

See accompanying Notes to Condensed Consolidated Financial Statements.

4

NVIDIA Corporation and Subsidiaries
Condensed Consolidated Balance Sheets
(In millions)
(Unaudited)
 	Apr 28, 2024		Jan 28, 2024
Assets			
Current assets:	 		 
Cash and cash equivalents	$	7,587 			$	7,280 	
Marketable securities	23,851 			18,704 	
Accounts receivable, net	12,365 			9,999 	
Inventories	5,864 			5,282 	
Prepaid expenses and other current assets	4,062 			3,080 	
Total current assets	53,729 			44,345 	
Property and equipment, net	4,006 			3,914 	
Operating lease assets	1,532 			1,346 	
Goodwill	4,453 			4,430 	
Intangible assets, net	986 			1,112 	
Deferred income tax assets	7,798 			6,081 	
Other assets	4,568 			4,500 	
Total assets	$	77,072 			$	65,728 	
Liabilities and Shareholders' Equity	 		 
Current liabilities:	 		 
Accounts payable	$	2,715 			$	2,699 	
Accrued and other current liabilities	11,258 			6,682 	
Short-term debt	1,250 			1,250 	
Total current liabilities	15,223 			10,631 	
Long-term debt	8,460 			8,459 	
Long-term operating lease liabilities	1,281 			1,119 	
Other long-term liabilities	2,966 			2,541 	
Total liabilities	27,930 			22,750 	
Commitments and contingencies - see Note 12			
Shareholders’ equity:	 		 
Preferred stock	— 			— 	
Common stock	2 			2 	
Additional paid-in capital	12,651 			13,132 	
Accumulated other comprehensive income (loss)	(109)			27 	
Retained earnings	36,598 			29,817 	
Total shareholders' equity	49,142 			42,978 	
Total liabilities and shareholders' equity	$	77,072 			$	65,728 	
 

See accompanying Notes to Condensed Consolidated Financial Statements.

5

NVIDIA Corporation and Subsidiaries
Condensed Consolidated Statements of Shareholders' Equity
For the Three Months Ended April 28, 2024 and April 30, 2023
(Unaudited)  
Common Stock
Outstanding		Additional Paid-in Capital		Accumulated Other Comprehensive Income (Loss)		Retained Earnings		Total Shareholders' Equity
Shares		Amount				
(In millions, except per share data)											
Balances, Jan 28, 2024	2,464 			$	2 			$	13,132 			$	27 			$	29,817 			$	42,978 	
Net income	— 			— 			— 			— 			14,881 			14,881 	
Other comprehensive loss	— 			— 			— 			(136)			— 			(136)	
Issuance of common stock from stock plans 	7 			— 			285 			— 			— 			285 	
Tax withholding related to vesting of restricted stock units	(2)			— 			(1,752)			— 			— 			(1,752)	
Shares repurchased	(10)			— 			(33)			— 			(8,002)			(8,035)	
Cash dividends declared and paid ($0.04 per common share)
— 			— 			— 			— 			(98)			(98)	
Stock-based compensation	— 			— 			1,019 			— 			— 			1,019 	
Balances, Apr 28, 2024	2,459 			$	2 			$	12,651 			$	(109)			$	36,598 			$	49,142 	
Balances, Jan 29, 2023	2,466 			$	2 			$	11,971 			$	(43)			$	10,171 			$	22,101 	
Net income	— 			— 			— 			— 			2,043 			2,043 	
Other comprehensive loss	— 			— 			— 			(7)			— 			(7)	
Issuance of common stock from stock plans 	9 			— 			246 			— 			— 			246 	
Tax withholding related to vesting of restricted stock units	(2)			— 			(507)			— 			— 			(507)	
Cash dividends declared and paid ($0.04 per common share)
— 			— 			— 			— 			(99)			(99)	
Stock-based compensation	— 			— 			743 			— 			— 			743 	
Balances, Apr 30, 2023	2,473 			$	2 			$	12,453 			$	(50)			$	12,115 			$	24,520 	
 
See accompanying Notes to Condensed Consolidated Financial Statements.
6

NVIDIA Corporation and Subsidiaries
Condensed Consolidated Statements of Cash Flows
(In millions)
(Unaudited) 
 	Three Months Ended
 	Apr 28, 2024		Apr 30, 2023
Cash flows from operating activities:			
Net income	$	14,881 			$	2,043 	
Adjustments to reconcile net income to net cash provided by operating activities:			
Stock-based compensation expense	1,011 			735 	
Depreciation and amortization	410 			384 	
Realized and unrealized (gains) losses on investments in non-affiliated entities, net	(69)			14 	
Deferred income taxes	(1,577)			(1,135)	
Other	(145)			(34)	
Changes in operating assets and liabilities, net of acquisitions:			
Accounts receivable	(2,366)			(252)	
Inventories	(577)			566 	
Prepaid expenses and other assets	(726)			(215)	
Accounts payable	(22)			11 	
Accrued and other current liabilities	4,202 			689 	
Other long-term liabilities	323 			105 	
Net cash provided by operating activities	15,345 			2,911 	
Cash flows from investing activities:			
Proceeds from maturities of marketable securities	4,004 			2,512 	
Proceeds from sales of marketable securities	149 			— 	
Purchases of marketable securities	(9,303)			(2,801)	
Purchases related to property and equipment and intangible assets	(369)			(248)	
Acquisitions, net of cash acquired	(39)			(83)	
Investments in non-affiliated entities	(135)			(221)	
Net cash used in investing activities	(5,693)			(841)	
Cash flows from financing activities:			
Proceeds related to employee stock plans	285 			246 	
Payments related to repurchases of common stock	(7,740)			— 	
Payments related to tax on restricted stock units	(1,752)			(507)	
Dividends paid	(98)			(99)	
Principal payments on property and equipment and intangible assets	(40)			(20)	
Net cash used in financing activities	(9,345)			(380)	
Change in cash and cash equivalents	307 			1,690 	
Cash and cash equivalents at beginning of period	7,280 			3,389 	
Cash and cash equivalents at end of period	$	7,587 			$	5,079 	
 
See accompanying Notes to Condensed Consolidated Financial Statements.
7
NVIDIA Corporation and Subsidiaries
Notes to Condensed Consolidated Financial Statements
(Unaudited)


Note 1 - Summary of Significant Accounting Policies
Basis of Presentation
The accompanying unaudited condensed consolidated financial statements were prepared in accordance with accounting principles generally accepted in the United States of America, or U.S. GAAP, for interim financial information and with the instructions to Form 10-Q and Article 10 of Securities and Exchange Commission, or SEC, Regulation S-X. The January 28, 2024 consolidated balance sheet was derived from our audited consolidated financial statements included in our Annual Report on Form 10-K for the fiscal year ended January 28, 2024, as filed with the SEC, but does not include all disclosures required by U.S. GAAP. In the opinion of management, all adjustments, consisting only of normal recurring adjustments considered necessary for a fair statement of results of operations and financial position, have been included. The results for the interim periods presented are not necessarily indicative of the results expected for any future period. The following information should be read in conjunction with the audited consolidated financial statements and notes thereto included in our Annual Report on Form 10-K for the fiscal year ended January 28, 2024. 
Significant Accounting Policies
There have been no material changes to our significant accounting policies disclosed in Note 1 - Organization and Summary of Significant Accounting Policies, of the Notes to the Consolidated Financial Statements included in our Annual Report on Form 10-K for the fiscal year ended January 28, 2024.
Fiscal Year
We operate on a 52- or 53-week year, ending on the last Sunday in January. Fiscal years 2025 and 2024 are both 52-week years. The first quarters of fiscal years 2025 and 2024 were both 13-week quarters.
Principles of Consolidation
Our condensed consolidated financial statements include the accounts of NVIDIA Corporation and our wholly-owned subsidiaries. All intercompany balances and transactions have been eliminated in consolidation.
Use of Estimates
The preparation of financial statements in conformity with U.S. GAAP requires management to make estimates and assumptions that affect the reported amounts of assets and liabilities and disclosures of contingent assets and liabilities at the date of the financial statements and the reported amounts of revenue and expenses during the reporting period. Actual results could differ materially from our estimates. On an on-going basis, we evaluate our estimates, including those related to revenue recognition, cash equivalents and marketable securities, accounts receivable, inventories and product purchase commitments, income taxes, goodwill, stock-based compensation, litigation, investigation and settlement costs, property, plant, and equipment, and other contingencies. These estimates are based on historical facts and various other assumptions that we believe are reasonable.
Recently Issued Accounting Pronouncements
Recent Accounting Pronouncements Not Yet Adopted
In November 2023, the Financial Accounting Standards Board, or FASB, issued a new accounting standard to provide for additional disclosures about significant expenses in operating segments. The standard is effective for our annual reporting starting with fiscal year 2025 and for interim period reporting starting in fiscal year 2026 retrospectively. We are currently evaluating the impact of this standard on our Consolidated Financial Statements.
In December 2023, the FASB issued a new accounting standard which provides for new and updated income tax disclosures, including disaggregation of rate reconciliation and income taxes paid. The standard is effective for annual periods beginning after December 15, 2024. Early adoption is permitted and should be applied prospectively, with retrospective application permitted. We expect to adopt this standard in our annual reporting starting with fiscal year 2026. We are currently evaluating the impact of this standard on our Consolidated Financial Statements.



8
NVIDIA Corporation and Subsidiaries
Notes to Condensed Consolidated Financial Statements (Continued)
(Unaudited)
Note 2 - Leases
Our lease obligations primarily consist of operating leases for our headquarters complex, domestic and international office facilities, and data center space, with lease periods expiring between fiscal years 2025 and 2035.
Future minimum lease payments under our non-cancelable operating leases as of April 28, 2024 were as follows:
Operating Lease Obligations
 	(In millions)
Fiscal Year:	 
2025 (excluding first quarter of fiscal year 2025)
$	221 	
2026	306 	
2027	290 	
2028	270 	
2029	236 	
2030 and thereafter
410 	
Total	1,733 	
Less imputed interest	206 	
Present value of net future minimum lease payments	1,527 	
Less short-term operating lease liabilities	246 	
Long-term operating lease liabilities	$	1,281 	
 
In addition, we have operating leases, primarily for our data centers, that are expected to commence during fiscal year 2025 with lease terms of 2 to 11 years for $923 million.
Operating lease expenses were $80 million and $59 million for the first quarter of fiscal years 2025 and 2024, respectively. Short-term and variable lease expenses for the first quarter of fiscal years 2025 and 2024 were not significant.
Other information related to leases was as follows:
Three Months Ended
Apr 28, 2024		Apr 30, 2023
 	(In millions)
Supplemental cash flows information			 
Operating cash flows used for operating leases	$	69 			$	61 	
Operating lease assets obtained in exchange for lease obligations	250 			106 	
 
As of April 28, 2024, our operating leases had a weighted average remaining lease term of 6.3 years and a weighted average discount rate of 3.89%. As of January 28, 2024, our operating leases had a weighted average remaining lease term of 6.1 years and a weighted average discount rate of 3.76%.
9
NVIDIA Corporation and Subsidiaries
Notes to Condensed Consolidated Financial Statements (Continued)
(Unaudited)
Note 3 - Stock-Based Compensation
Our stock-based compensation expense is associated with restricted stock units, or RSUs, performance stock units that are based on our corporate financial performance targets, or PSUs, performance stock units that are based on market conditions, or market-based PSUs, and our employee stock purchase plan, or ESPP.
Our Condensed Consolidated Statements of Income include stock-based compensation expense, net of amounts capitalized into inventory and subsequently recognized to cost of revenue, as follows:
 	Three Months Ended
 	Apr 28, 2024		Apr 30, 2023
(In millions)
Cost of revenue	$	36 			$	27 	
Research and development	727 			524 	
Sales, general and administrative	248 			184 	
Total	$	1,011 			$	735 	
 
Equity Award Activity
The following is a summary of our equity award transactions under our equity incentive plans:
RSUs, PSUs, and Market-based PSUs Outstanding
 	Number of Shares		Weighted Average Grant-Date Fair Value Per Share
(In millions, except per share data)
Balances, Jan 28, 2024	37 			$	245.94 	
Granted	7 			$	801.79 	
Vested	(6)			$	176.59 	
Balances, Apr 28, 2024	38 			$	361.45 	
 
As of April 28, 2024, there was $13.2 billion of aggregate unearned stock-based compensation expense. This amount is expected to be recognized over a weighted average period of 2.6 years for RSUs, PSUs, and market-based PSUs, and 0.8 years for ESPP.
Note 4 - Net Income Per Share
The following is a reconciliation of the denominator of the basic and diluted net income per share computations for the periods presented:
 

"""

# Create an instance of the OpenAIChat class
llm = OpenAIChat(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    temperature=0.1,
)

# Initialize the director agent
# Initialize the director agent
director = Agent(
    agent_name="Director",
    system_prompt="Directs the tasks for the accountants",
    llm=llm,
    max_loops=1,
    dashboard=False,
    state_save_file_type="json",
    saved_state_path="director.json",
)

# Initialize accountant 1
accountant1 = Agent(
    agent_name="Accountant1",
    system_prompt="Prepares financial statements",
    llm=llm,
    max_loops=1,
    dashboard=False,
    state_save_file_type="json",
    saved_state_path="accountant1.json",
)

# Initialize accountant 2
accountant2 = Agent(
    agent_name="Accountant2",
    system_prompt="Audits financial records",
    llm=llm,
    max_loops=1,
    dashboard=False,
    state_save_file_type="json",
    saved_state_path="accountant2.json",
)

# Initialize 8 more specialized agents
balance_sheet_analyzer = Agent(
    agent_name="BalanceSheetAnalyzer",
    system_prompt="Analyzes balance sheets",
    llm=llm,
    max_loops=1,
    dashboard=False,
    state_save_file_type="json",
    saved_state_path="balance_sheet_analyzer.json",
)

income_statement_analyzer = Agent(
    agent_name="IncomeStatementAnalyzer",
    system_prompt="Analyzes income statements",
    llm=llm,
    max_loops=1,
    dashboard=False,
    state_save_file_type="json",
    saved_state_path="income_statement_analyzer.json",
)

cash_flow_analyzer = Agent(
    agent_name="CashFlowAnalyzer",
    system_prompt="Analyzes cash flow statements",
    llm=llm,
    max_loops=1,
    dashboard=False,
    state_save_file_type="json",
    saved_state_path="cash_flow_analyzer.json",
)

financial_ratio_calculator = Agent(
    agent_name="FinancialRatioCalculator",
    system_prompt="Calculates financial ratios",
    llm=llm,
    max_loops=1,
    dashboard=False,
    state_save_file_type="json",
    saved_state_path="financial_ratio_calculator.json",
)

tax_preparer = Agent(
    agent_name="TaxPreparer",
    system_prompt="Prepares tax returns",
    llm=llm,
    max_loops=1,
    dashboard=False,
    state_save_file_type="json",
    saved_state_path="tax_preparer.json",
)

payroll_processor = Agent(
    agent_name="PayrollProcessor",
    system_prompt="Processes payroll",
    llm=llm,
    max_loops=1,
    dashboard=False,
    state_save_file_type="json",
    saved_state_path="payroll_processor.json",
)

inventory_manager = Agent(
    agent_name="InventoryManager",
    system_prompt="Manages inventory",
    llm=llm,
    max_loops=1,
    dashboard=False,
    state_save_file_type="json",
    saved_state_path="inventory_manager.json",
)

budget_planner = Agent(
    agent_name="BudgetPlanner",
    system_prompt="Plans budgets",
    llm=llm,
    max_loops=1,
    dashboard=False,
    state_save_file_type="json",
    saved_state_path="budget_planner.json",
)

# Create a list of agents
agents = [
    director,
    balance_sheet_analyzer,
    income_statement_analyzer,
]

# Swarm
swarm = MixtureOfAgents(
    name="Mixture of Accountants",
    agents=agents,
    layers=3,
    final_agent=director,
    agents_per_layer=3,
)


# Run the swarm
out = swarm.run(
    f"Analyze the following Nvidia financial data and locate unnecessary expenditures: {SEC_DATA}"
)
print(out)
