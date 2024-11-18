"""
Advanced Financial Backtesting System
-----------------------------------
A comprehensive system for backtesting trading strategies using the Swarms framework,
real-time data from Yahoo Finance, and AI-driven decision making.

Features:
- Type-safe implementation with comprehensive type hints
- Detailed logging with Loguru
- Real-time data fetching from Yahoo Finance
- Advanced technical analysis
- Performance metrics and visualization
- AI-driven trading decisions using Swarms framework
"""

import os
from datetime import datetime
from typing import Dict, List, TypedDict
from dataclasses import dataclass
import pandas as pd
import numpy as np
import yfinance as yf
from swarms import Agent
from swarm_models import OpenAIChat
from dotenv import load_dotenv
from loguru import logger

# Configure logging
logger.add(
    "backtester_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


# Type definitions
class TradeAction(TypedDict):
    date: datetime
    action: str
    symbol: str
    quantity: float
    price: float
    cash: float
    commission: float


class PortfolioMetrics(TypedDict):
    total_return: float
    total_trades: int
    total_commission: float
    final_cash: float
    sharpe_ratio: float
    max_drawdown: float


@dataclass
class TechnicalIndicators:
    sma_20: float
    sma_50: float
    rsi: float
    macd: float
    signal_line: float
    volume: int


class FinancialData:
    """
    Handles financial data operations using Yahoo Finance API

    Attributes:
        cache (Dict): Cache for storing downloaded data
    """

    def __init__(self) -> None:
        self.cache: Dict[str, pd.DataFrame] = {}

    @logger.catch
    def get_historical_prices(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetches historical price data from Yahoo Finance

        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame containing historical price data
        """
        logger.info(
            f"Fetching data for {symbol} from {start_date} to {end_date}"
        )

        if symbol not in self.cache:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                df["symbol"] = symbol
                df.index.name = "date"
                df.reset_index(inplace=True)
                self.cache[symbol] = df
                logger.success(
                    f"Successfully downloaded data for {symbol}"
                )
            except Exception as e:
                logger.error(
                    f"Error fetching data for {symbol}: {str(e)}"
                )
                raise

        return self.cache[symbol]

    @logger.catch
    def get_technical_indicators(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates technical indicators for analysis

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with added technical indicators
        """
        logger.info("Calculating technical indicators")
        df = df.copy()

        try:
            # Calculate moving averages
            df["SMA_20"] = df["Close"].rolling(window=20).mean()
            df["SMA_50"] = df["Close"].rolling(window=50).mean()

            # Calculate RSI
            delta = df["Close"].diff()
            gain = (
                (delta.where(delta > 0, 0)).rolling(window=14).mean()
            )
            loss = (
                (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            )
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

            # Calculate MACD
            exp1 = df["Close"].ewm(span=12, adjust=False).mean()
            exp2 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = exp1 - exp2
            df["Signal_Line"] = (
                df["MACD"].ewm(span=9, adjust=False).mean()
            )

            logger.success(
                "Successfully calculated technical indicators"
            )
            return df

        except Exception as e:
            logger.error(
                f"Error calculating technical indicators: {str(e)}"
            )
            raise


class Portfolio:
    """
    Manages portfolio positions and tracks performance

    Attributes:
        initial_cash: Starting capital
        cash: Current cash balance
        positions: Current stock positions
        history: Trade history
        trade_count: Number of trades executed
    """

    def __init__(self, initial_cash: float = 100000.0) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}
        self.history: List[TradeAction] = []
        self.trade_count = 0
        logger.info(
            f"Initialized portfolio with ${initial_cash:,.2f}"
        )

    @logger.catch
    def execute_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
        date: datetime,
    ) -> None:
        """
        Executes a trade and updates portfolio state

        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            price: Trade price
            quantity: Number of shares
            date: Trade date
        """
        commission = 1.0  # $1 per trade commission

        try:
            if action == "BUY":
                cost = (price * quantity) + commission
                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[symbol] = (
                        self.positions.get(symbol, 0) + quantity
                    )
                    self.trade_count += 1
                    logger.info(
                        f"Bought {quantity} shares of {symbol} at ${price:.2f}"
                    )
            elif action == "SELL":
                if (
                    symbol in self.positions
                    and self.positions[symbol] >= quantity
                ):
                    self.cash += (price * quantity) - commission
                    self.positions[symbol] -= quantity
                    if self.positions[symbol] == 0:
                        del self.positions[symbol]
                    self.trade_count += 1
                    logger.info(
                        f"Sold {quantity} shares of {symbol} at ${price:.2f}"
                    )

            self.history.append(
                {
                    "date": date,
                    "action": action,
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "cash": self.cash,
                    "commission": commission,
                }
            )

        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            raise

    def get_metrics(self) -> PortfolioMetrics:
        """
        Calculates portfolio performance metrics

        Returns:
            Dictionary containing performance metrics
        """
        try:
            df = pd.DataFrame(self.history)
            if len(df) == 0:
                return {
                    "total_return": 0.0,
                    "total_trades": 0,
                    "total_commission": 0.0,
                    "final_cash": self.initial_cash,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                }

            portfolio_values = df["cash"].values
            returns = (
                np.diff(portfolio_values) / portfolio_values[:-1]
            )

            sharpe_ratio = (
                np.sqrt(252) * np.mean(returns) / np.std(returns)
                if len(returns) > 0
                else 0
            )
            max_drawdown = np.min(
                np.minimum.accumulate(portfolio_values)
                / np.maximum.accumulate(portfolio_values)
                - 1
            )

            metrics: PortfolioMetrics = {
                "total_return": (
                    (self.cash - self.initial_cash)
                    / self.initial_cash
                )
                * 100,
                "total_trades": self.trade_count,
                "total_commission": self.trade_count * 1.0,
                "final_cash": self.cash,
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown * 100),
            }

            logger.info("Successfully calculated portfolio metrics")
            return metrics

        except Exception as e:
            logger.error(
                f"Error calculating portfolio metrics: {str(e)}"
            )
            raise


class FinancialAgent:
    """
    AI Agent for making trading decisions using the Swarms framework

    Attributes:
        model: OpenAI chat model instance
        agent: Swarms agent instance
    """

    def __init__(self, api_key: str) -> None:
        logger.info("Initializing Financial Agent")

        self.model = OpenAIChat(
            openai_api_key=api_key,
            model_name="gpt-4-0125-preview",
            temperature=0.1,
        )

        self.agent = Agent(
            agent_name="Financial-Trading-Agent",
            system_prompt="""You are an AI trading agent. Analyze the provided price data and technical indicators to make trading decisions. 
            Output only one of these decisions: BUY, SELL, or HOLD. Consider the following in your analysis:
            1. Trend direction using moving averages (SMA_20 and SMA_50)
            2. RSI for overbought/oversold conditions (>70 overbought, <30 oversold)
            3. MACD crossovers and momentum
            4. Recent price action and volume
            
            Provide your decision in a single word: BUY, SELL, or HOLD.""",
            llm=self.model,
            max_loops=1,
            autosave=True,
            dashboard=False,
            verbose=True,
        )

    @logger.catch
    def make_decision(self, price_data: pd.DataFrame) -> str:
        """
        Makes trading decision based on price data and technical indicators

        Args:
            price_data: DataFrame containing price and indicator data

        Returns:
            Trading decision: 'BUY', 'SELL', or 'HOLD'
        """
        try:
            latest_data = price_data.tail(1).to_dict("records")[0]

            prompt = f"""
            Current Market Data:
            Price: ${latest_data['Close']:.2f}
            SMA_20: ${latest_data['SMA_20']:.2f}
            SMA_50: ${latest_data['SMA_50']:.2f}
            RSI: {latest_data['RSI']:.2f}
            MACD: {latest_data['MACD']:.2f}
            Signal Line: {latest_data['Signal_Line']:.2f}
            Volume: {latest_data['Volume']}
            
            Based on this data, what is your trading decision?
            """

            decision = self.agent.run(prompt)
            decision = decision.strip().upper()

            if decision not in ["BUY", "SELL", "HOLD"]:
                logger.warning(
                    f"Invalid decision '{decision}', defaulting to HOLD"
                )
                decision = "HOLD"

            logger.info(f"Agent decision: {decision}")
            return decision

        except Exception as e:
            logger.error(f"Error making trading decision: {str(e)}")
            raise


class Backtester:
    """
    Runs trading strategy backtests and analyzes performance

    Attributes:
        agent: Trading agent instance
        portfolio: Portfolio instance
        results: List of backtest results
    """

    def __init__(
        self, agent: FinancialAgent, portfolio: Portfolio
    ) -> None:
        self.agent = agent
        self.portfolio = portfolio
        self.results: List[Dict] = []
        logger.info("Initialized Backtester")

    @logger.catch
    def run_backtest(
        self, price_data: pd.DataFrame, trade_size: float = 100
    ) -> None:
        """
        Runs backtest simulation

        Args:
            price_data: Historical price data
            trade_size: Number of shares per trade
        """
        logger.info("Starting backtest")

        try:
            df = FinancialData().get_technical_indicators(price_data)
            df = df.dropna()

            for i in range(len(df)):
                current_data = df.iloc[i]
                current_price = current_data["Close"]
                current_date = current_data["date"]

                decision = self.agent.make_decision(
                    df.iloc[max(0, i - 10) : i + 1]
                )

                if decision == "BUY":
                    self.portfolio.execute_trade(
                        symbol=current_data["symbol"],
                        action="BUY",
                        price=current_price,
                        quantity=trade_size,
                        date=current_date,
                    )
                elif decision == "SELL":
                    self.portfolio.execute_trade(
                        symbol=current_data["symbol"],
                        action="SELL",
                        price=current_price,
                        quantity=trade_size,
                        date=current_date,
                    )

                portfolio_value = self.portfolio.get_metrics()[
                    "final_cash"
                ]

                self.results.append(
                    {
                        "date": current_date,
                        "price": current_price,
                        "decision": decision,
                        "portfolio_value": portfolio_value,
                        "SMA_20": current_data["SMA_20"],
                        "SMA_50": current_data["SMA_50"],
                        "RSI": current_data["RSI"],
                        "MACD": current_data["MACD"],
                    }
                )

            logger.success("Backtest completed successfully")

        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            raise

    def get_results(self) -> pd.DataFrame:
        """
        Returns backtest results as DataFrame
        """
        return pd.DataFrame(self.results)


def main() -> None:
    """
    Main function to run the backtesting system
    """
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found in environment variables"
            )

        # Initialize components
        data_provider = FinancialData()
        agent = FinancialAgent(api_key)
        portfolio = Portfolio(initial_cash=100000.0)
        backtester = Backtester(agent, portfolio)

        # Get historical data
        symbol = "AAPL"
        start_date = "2023-01-01"
        end_date = "2023-12-31"

        logger.info(
            f"Starting backtest for {symbol} from {start_date} to {end_date}"
        )

        price_data = data_provider.get_historical_prices(
            symbol, start_date, end_date
        )
        backtester.run_backtest(price_data)

        # Get and display results
        results = backtester.get_results()
        metrics = portfolio.get_metrics()

        logger.info("Backtest Results:")
        logger.info(f"Initial Portfolio Value: ${100000:.2f}")
        logger.info(
            f"Final Portfolio Value: ${metrics['final_cash']:.2f}"
        )
        logger.info(f"Total Return: {metrics['total_return']:.2f}%")
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(
            f"Total Commission: ${metrics['total_commission']:.2f}"
        )
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    main()
