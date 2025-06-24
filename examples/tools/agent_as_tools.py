import json
import requests
from swarms import Agent


def create_python_file(code: str, filename: str) -> str:
    """Create a Python file with the given code and execute it using Python 3.12.

    This function takes a string containing Python code, writes it to a file, and executes it
    using Python 3.12 via subprocess. The file will be created in the current working directory.
    If a file with the same name already exists, it will be overwritten.

    Args:
        code (str): The Python code to write to the file. This should be valid Python 3.12 code.
        filename (str): The name of the file to create and execute.

    Returns:
        str: A detailed message indicating the file was created and the execution result.

    Raises:
        IOError: If there are any issues writing to the file.
        subprocess.SubprocessError: If there are any issues executing the file.

    Example:
        >>> code = "print('Hello, World!')"
        >>> result = create_python_file(code, "test.py")
        >>> print(result)
        'Python file created successfully. Execution result: Hello, World!'
    """
    import subprocess
    import os
    import datetime

    # Get current timestamp for logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write the code to file
    with open(filename, "w") as f:
        f.write(code)

    # Get file size and permissions
    file_stats = os.stat(filename)
    file_size = file_stats.st_size
    file_permissions = oct(file_stats.st_mode)[-3:]

    # Execute the file using Python 3.12 and capture output
    try:
        result = subprocess.run(
            ["python3.12", filename],
            capture_output=True,
            text=True,
            check=True,
        )

        # Create detailed response
        response = f"""
File Creation Details:
----------------------
Timestamp: {timestamp}
Filename: {filename}
File Size: {file_size} bytes
File Permissions: {file_permissions}
Location: {os.path.abspath(filename)}

Execution Details:
-----------------
Exit Code: {result.returncode}
Execution Time: {result.returncode} seconds

Output:
-------
{result.stdout}

Error Output (if any):
--------------------
{result.stderr}
"""
        return response
    except subprocess.CalledProcessError as e:
        error_response = f"""
File Creation Details:
----------------------
Timestamp: {timestamp}
Filename: {filename}
File Size: {file_size} bytes
File Permissions: {file_permissions}
Location: {os.path.abspath(filename)}

Execution Error:
---------------
Exit Code: {e.returncode}
Error Message: {e.stderr}

Command Output:
-------------
{e.stdout}
"""
        return error_response


def update_python_file(code: str, filename: str) -> str:
    """Update an existing Python file with new code and execute it using Python 3.12.

    This function takes a string containing Python code and updates an existing Python file.
    If the file doesn't exist, it will be created. The file will be executed using Python 3.12.

    Args:
        code (str): The Python code to write to the file. This should be valid Python 3.12 code.
        filename (str): The name of the file to update and execute.

    Returns:
        str: A detailed message indicating the file was updated and the execution result.

    Raises:
        IOError: If there are any issues writing to the file.
        subprocess.SubprocessError: If there are any issues executing the file.

    Example:
        >>> code = "print('Updated code!')"
        >>> result = update_python_file(code, "my_script.py")
        >>> print(result)
        'Python file updated successfully. Execution result: Updated code!'
    """
    import subprocess
    import os
    import datetime

    # Get current timestamp for logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if file exists and get its stats
    file_exists = os.path.exists(filename)
    if file_exists:
        old_stats = os.stat(filename)
        old_size = old_stats.st_size
        old_permissions = oct(old_stats.st_mode)[-3:]

    # Write the code to file
    with open(filename, "w") as f:
        f.write(code)

    # Get new file stats
    new_stats = os.stat(filename)
    new_size = new_stats.st_size
    new_permissions = oct(new_stats.st_mode)[-3:]

    # Execute the file using Python 3.12 and capture output
    try:
        result = subprocess.run(
            ["python3.12", filename],
            capture_output=True,
            text=True,
            check=True,
        )

        # Create detailed response
        response = f"""
File Update Details:
-------------------
Timestamp: {timestamp}
Filename: {filename}
Previous Status: {'Existed' if file_exists else 'Did not exist'}
Previous Size: {old_size if file_exists else 'N/A'} bytes
Previous Permissions: {old_permissions if file_exists else 'N/A'}
New Size: {new_size} bytes
New Permissions: {new_permissions}
Location: {os.path.abspath(filename)}

Execution Details:
-----------------
Exit Code: {result.returncode}
Execution Time: {result.returncode} seconds

Output:
-------
{result.stdout}

Error Output (if any):
--------------------
{result.stderr}
"""
        return response
    except subprocess.CalledProcessError as e:
        error_response = f"""
        File Update Details:
        -------------------
        Timestamp: {timestamp}
        Filename: {filename}
        Previous Status: {'Existed' if file_exists else 'Did not exist'}
        Previous Size: {old_size if file_exists else 'N/A'} bytes
        Previous Permissions: {old_permissions if file_exists else 'N/A'}
        New Size: {new_size} bytes
        New Permissions: {new_permissions}
        Location: {os.path.abspath(filename)}

        Execution Error:
        ---------------
        Exit Code: {e.returncode}
        Error Message: {e.stderr}

        Command Output:
        -------------
        {e.stdout}
        """
        return error_response


def run_quant_trading_agent(task: str) -> str:
    """Run a quantitative trading agent to analyze and execute trading strategies.

    This function initializes and runs a specialized quantitative trading agent that can:
    - Develop and backtest trading strategies
    - Analyze market data for alpha opportunities
    - Implement risk management frameworks
    - Optimize portfolio allocations
    - Conduct quantitative research
    - Monitor market microstructure
    - Evaluate trading system performance

    Args:
        task (str): The specific trading task or analysis to perform

    Returns:
        str: The agent's response or analysis results

    Example:
        >>> result = run_quant_trading_agent("Analyze SPY ETF for mean reversion opportunities")
        >>> print(result)
    """
    # Initialize the agent
    agent = Agent(
        agent_name="Quantitative-Trading-Agent",
        agent_description="Advanced quantitative trading and algorithmic analysis agent",
        system_prompt="""You are an expert quantitative trading agent with deep expertise in:
        - Algorithmic trading strategies and implementation
        - Statistical arbitrage and market making
        - Risk management and portfolio optimization
        - High-frequency trading systems
        - Market microstructure analysis
        - Quantitative research methodologies
        - Financial mathematics and stochastic processes
        - Machine learning applications in trading
        
        Your core responsibilities include:
        1. Developing and backtesting trading strategies
        2. Analyzing market data and identifying alpha opportunities
        3. Implementing risk management frameworks
        4. Optimizing portfolio allocations
        5. Conducting quantitative research
        6. Monitoring market microstructure
        7. Evaluating trading system performance
        
        You maintain strict adherence to:
        - Mathematical rigor in all analyses
        - Statistical significance in strategy development
        - Risk-adjusted return optimization
        - Market impact minimization
        - Regulatory compliance
        - Transaction cost analysis
        - Performance attribution
        
        You communicate in precise, technical terms while maintaining clarity for stakeholders.""",
        max_loops=2,
        model_name="claude-3-5-sonnet-20240620",
        tools=[
            create_python_file,
            update_python_file,
            backtest_summary,
        ],
    )

    out = agent.run(task)
    return out


def backtest_summary(report: str) -> str:
    """Generate a summary of a backtest report, but only if the backtest was profitable.

    This function should only be used when the backtest results show a positive return.
    Using this function for unprofitable backtests may lead to misleading conclusions.

    Args:
        report (str): The backtest report containing performance metrics

    Returns:
        str: A formatted summary of the backtest report

    Example:
        >>> result = backtest_summary("Total Return: +15.2%, Sharpe: 1.8")
        >>> print(result)
        'The backtest report is: Total Return: +15.2%, Sharpe: 1.8'
    """
    return f"The backtest report is: {report}"


def get_coin_price(coin_id: str, vs_currency: str) -> str:
    """
    Get the current price of a specific cryptocurrency.

    Args:
        coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')
        vs_currency (str, optional): The target currency. Defaults to "usd".

    Returns:
        str: JSON formatted string containing the coin's current price and market data

    Raises:
        requests.RequestException: If the API request fails

    Example:
        >>> result = get_coin_price("bitcoin")
        >>> print(result)
        {"bitcoin": {"usd": 45000, "usd_market_cap": 850000000000, ...}}
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": vs_currency,
            "include_market_cap": True,
            "include_24hr_vol": True,
            "include_24hr_change": True,
            "include_last_updated_at": True,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        return json.dumps(data, indent=2)

    except requests.RequestException as e:
        return json.dumps(
            {
                "error": f"Failed to fetch price for {coin_id}: {str(e)}"
            }
        )
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def run_crypto_quant_agent(task: str) -> str:
    """
    Run a crypto quantitative trading agent with specialized tools for cryptocurrency market analysis.

    This function initializes and runs a quantitative trading agent specifically designed for
    cryptocurrency markets. The agent is equipped with tools for price fetching and can perform
    various quantitative analyses including algorithmic trading strategy development, risk management,
    and market microstructure analysis.

    Args:
        task (str): The task or query to be processed by the crypto quant agent.

    Returns:
        str: The agent's response to the given task.

    Example:
        >>> response = run_crypto_quant_agent("Analyze the current market conditions for Bitcoin")
        >>> print(response)
        "Based on current market analysis..."
    """
    # Initialize the agent with expanded tools
    quant_agent = Agent(
        agent_name="Crypto-Quant-Agent",
        agent_description="Advanced quantitative trading agent specializing in cryptocurrency markets with algorithmic analysis capabilities",
        system_prompt="""You are an expert quantitative trading agent specializing in cryptocurrency markets. Your capabilities include:
        - Algorithmic trading strategy development and backtesting
        - Statistical arbitrage and market making for crypto assets
        - Risk management and portfolio optimization for digital assets
        - High-frequency trading system design for crypto markets
        - Market microstructure analysis of crypto exchanges
        - Quantitative research methodologies for crypto assets
        - Financial mathematics and stochastic processes
        - Machine learning applications in crypto trading
        
        You maintain strict adherence to:
        - Mathematical rigor in all analyses
        - Statistical significance in strategy development
        - Risk-adjusted return optimization
        - Market impact minimization
        - Regulatory compliance
        - Transaction cost analysis
        - Performance attribution
        
        You communicate in precise, technical terms while maintaining clarity for stakeholders.""",
        max_loops=1,
        max_tokens=4096,
        model_name="gpt-4.1-mini",
        dynamic_temperature_enabled=True,
        output_type="final",
        tools=[
            get_coin_price,
        ],
    )

    return quant_agent.run(task)


# Initialize the agent
agent = Agent(
    agent_name="Director-Agent",
    agent_description="Strategic director and project management agent",
    system_prompt="""You are an expert Director Agent with comprehensive capabilities in:
    - Strategic planning and decision making
    - Project management and coordination
    - Resource allocation and optimization
    - Team leadership and delegation
    - Risk assessment and mitigation
    - Stakeholder management
    - Process optimization
    - Quality assurance
    
    Your core responsibilities include:
    1. Developing and executing strategic initiatives
    2. Coordinating cross-functional projects
    3. Managing resource allocation
    4. Setting and tracking KPIs
    5. Ensuring project deliverables
    6. Risk management and mitigation
    7. Stakeholder communication
    
    You maintain strict adherence to:
    - Best practices in project management
    - Data-driven decision making
    - Clear communication protocols
    - Quality standards
    - Timeline management
    - Budget constraints
    - Regulatory compliance
    
    You communicate with clarity and authority while maintaining professionalism and ensuring all stakeholders are aligned.""",
    max_loops=1,
    model_name="gpt-4o-mini",
    output_type="final",
    interactive=False,
    tools=[run_quant_trading_agent],
)

out = agent.run(
    """
    Please call the quantitative trading agent to generate Python code for an Bitcoin backtest using the CoinGecko API.
    Provide a comprehensive description of the backtest methodology and trading strategy.
    Consider the API limitations of CoinGecko and utilize only free, open-source libraries that don't require API keys. Use the requests library to fetch the data. Create a specialized strategy for the backtest focused on the orderbook and other data for price action.
    The goal is to create a backtest that can predict the price action of the coin based on the orderbook and other data.
    Maximize the profit of the backtest. Please use the OKX price API for the orderbook and other data. Be very explicit in your implementation.
    Be very precise with the instructions you give to the agent and tell it to a 400 lines of good code.
"""
)
print(out)
