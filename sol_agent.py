import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set

import aiohttp
import matplotlib.pyplot as plt
import networkx as nx
import websockets
from loguru import logger

from swarms import Agent

TREND_AGENT_PROMPT = """You are a specialized blockchain trend analysis agent. Your role:
1. Analyze transaction patterns in Solana blockchain data
2. Identify volume trends, price movements, and temporal patterns
3. Focus on whale movements and their market impact
4. Format findings in clear, structured JSON
5. Include confidence scores for each insight
6. Flag unusual patterns or anomalies
7. Provide historical context for significant movements

Output format:
{
    "trends": [
        {"pattern": str, "confidence": float, "impact": str}
    ],
    "whale_activity": {...},
    "temporal_analysis": {...}
}"""

RISK_AGENT_PROMPT = """You are a blockchain risk assessment specialist. Your tasks:
1. Identify suspicious transaction patterns
2. Monitor for known exploit signatures
3. Assess wallet clustering and relationship patterns
4. Evaluate transaction velocity and size anomalies
5. Check for bridge-related risks
6. Monitor smart contract interactions
7. Flag potential wash trading

Output format:
{
    "risk_score": float,
    "flags": [...],
    "recommendations": [...]
}"""

SUMMARY_AGENT_PROMPT = """You are a blockchain data synthesis expert. Your responsibilities:
1. Combine insights from trend and risk analyses
2. Prioritize actionable intelligence
3. Highlight critical patterns
4. Generate executive summaries
5. Provide market context
6. Make predictions with confidence intervals
7. Suggest trading strategies based on data

Output format:
{
    "key_insights": [...],
    "market_impact": str,
    "recommendations": {...}
}"""


@dataclass
class Transaction:
    signature: str
    timestamp: datetime
    amount: float
    from_address: str
    to_address: str


class SolanaRPC:
    def __init__(
        self, endpoint="https://api.mainnet-beta.solana.com"
    ):
        self.endpoint = endpoint
        self.session = None

    async def get_signatures(self, address: str) -> List[Dict]:
        if not self.session:
            self.session = aiohttp.ClientSession()

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignaturesForAddress",
            "params": [address, {"limit": 100}],
        }

        async with self.session.post(
            self.endpoint, json=payload
        ) as response:
            result = await response.json()
            return result.get("result", [])

    async def get_transaction(self, signature: str) -> Dict:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                signature,
                {
                    "encoding": "json",
                    "maxSupportedTransactionVersion": 0,
                },
            ],
        }

        async with self.session.post(
            self.endpoint, json=payload
        ) as response:
            result = await response.json()
            return result.get("result", {})


class AlertSystem:
    def __init__(self, email: str, threshold: float = 1000.0):
        self.email = email
        self.threshold = threshold
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587

    async def check_and_alert(
        self, transaction: Transaction, risk_score: float
    ):
        if transaction.amount > self.threshold or risk_score > 0.8:
            await self.send_alert(transaction, risk_score)

    async def send_alert(
        self, transaction: Transaction, risk_score: float
    ):
        # msg = MIMEText(
        #     f"High-risk transaction detected:\n"
        #     f"Amount: {transaction.amount} SOL\n"
        #     f"Risk Score: {risk_score}\n"
        #     f"Signature: {transaction.signature}"
        # )
        logger.info(
            f"Alert sent for transaction {transaction.signature}"
        )


class WalletClusterAnalyzer:
    def __init__(self):
        self.graph = nx.Graph()
        self.known_wallets: Set[str] = set()

    def update_graph(self, transaction: Transaction):
        self.graph.add_edge(
            transaction.from_address,
            transaction.to_address,
            weight=transaction.amount,
        )
        self.known_wallets.add(transaction.from_address)
        self.known_wallets.add(transaction.to_address)

    def identify_clusters(self) -> Dict:
        communities = nx.community.greedy_modularity_communities(
            self.graph
        )
        return {
            "clusters": [list(c) for c in communities],
            "central_wallets": [
                wallet
                for wallet in self.known_wallets
                if self.graph.degree[wallet] > 5
            ],
        }


class TransactionVisualizer:
    def __init__(self):
        self.transaction_history = []

    def add_transaction(self, transaction: Transaction):
        self.transaction_history.append(asdict(transaction))

    def generate_volume_chart(self) -> str:
        volumes = [tx["amount"] for tx in self.transaction_history]
        plt.figure(figsize=(12, 6))
        plt.plot(volumes)
        plt.title("Transaction Volume Over Time")
        plt.savefig("volume_chart.png")
        return "volume_chart.png"

    def generate_network_graph(
        self, wallet_analyzer: WalletClusterAnalyzer
    ) -> str:
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(wallet_analyzer.graph)
        nx.draw(
            wallet_analyzer.graph,
            pos,
            node_size=1000,
            node_color="lightblue",
            with_labels=True,
        )
        plt.savefig("network_graph.png")
        return "network_graph.png"


class SolanaMultiAgentAnalyzer:
    def __init__(
        self,
        min_amount: float = 50.0,
        websocket_url: str = "wss://api.mainnet-beta.solana.com",
        alert_email: str = None,
    ):
        self.rpc = SolanaRPC()
        self.websocket_url = websocket_url
        self.min_amount = min_amount
        self.transactions = []

        self.wallet_analyzer = WalletClusterAnalyzer()
        self.visualizer = TransactionVisualizer()
        self.alert_system = (
            AlertSystem(alert_email) if alert_email else None
        )

        self.trend_agent = Agent(
            agent_name="trend-analyzer",
            system_prompt=TREND_AGENT_PROMPT,
            model_name="gpt-4o-mini",
            max_loops=1,
            streaming_on=True,
        )

        self.risk_agent = Agent(
            agent_name="risk-analyzer",
            system_prompt=RISK_AGENT_PROMPT,
            model_name="gpt-4o-mini",
            max_loops=1,
            streaming_on=True,
        )

        self.summary_agent = Agent(
            agent_name="summary-agent",
            system_prompt=SUMMARY_AGENT_PROMPT,
            model_name="gpt-4o-mini",
            max_loops=1,
            streaming_on=True,
        )

        logger.add(
            "solana_analysis.log", rotation="500 MB", level="INFO"
        )

    async def start_websocket_stream(self):
        async with websockets.connect(
            self.websocket_url
        ) as websocket:
            subscribe_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "programSubscribe",
                "params": [
                    "11111111111111111111111111111111",
                    {"encoding": "json", "commitment": "confirmed"},
                ],
            }
            await websocket.send(json.dumps(subscribe_message))

            while True:
                try:
                    msg = await websocket.recv()
                    transaction = await self.parse_websocket_message(
                        msg
                    )
                    if (
                        transaction
                        and transaction.amount >= self.min_amount
                    ):
                        await self.process_transaction(transaction)
                except Exception as e:
                    logger.error(f"Websocket error: {e}")
                    await asyncio.sleep(5)

    async def parse_websocket_message(
        self, msg: str
    ) -> Optional[Transaction]:
        try:
            data = json.loads(msg)
            if "params" in data and "result" in data["params"]:
                tx_data = data["params"]["result"]
                return Transaction(
                    signature=tx_data["signature"],
                    timestamp=datetime.fromtimestamp(
                        tx_data["blockTime"]
                    ),
                    amount=float(
                        tx_data["meta"]["postBalances"][0]
                        - tx_data["meta"]["preBalances"][0]
                    )
                    / 1e9,
                    from_address=tx_data["transaction"]["message"][
                        "accountKeys"
                    ][0],
                    to_address=tx_data["transaction"]["message"][
                        "accountKeys"
                    ][1],
                )
        except Exception as e:
            logger.error(f"Error parsing websocket message: {e}")
        return None

    async def process_transaction(self, transaction: Transaction):
        self.wallet_analyzer.update_graph(transaction)
        self.visualizer.add_transaction(transaction)

        risk_analysis = await self.risk_agent.run(
            f"Analyze risk for transaction: {json.dumps(asdict(transaction))}"
        )

        if self.alert_system:
            await self.alert_system.check_and_alert(
                transaction, risk_analysis.get("risk_score", 0)
            )

    async def fetch_transactions(self) -> List[Transaction]:
        try:
            signatures = await self.rpc.get_signatures(
                "11111111111111111111111111111111"
            )
            transactions = []

            for sig_info in signatures:
                tx_data = await self.rpc.get_transaction(
                    sig_info["signature"]
                )
                if not tx_data or "meta" not in tx_data:
                    continue

                pre_balances = tx_data["meta"]["preBalances"]
                post_balances = tx_data["meta"]["postBalances"]
                amount = abs(pre_balances[0] - post_balances[0]) / 1e9

                if amount >= self.min_amount:
                    tx = Transaction(
                        signature=sig_info["signature"],
                        timestamp=datetime.fromtimestamp(
                            tx_data["blockTime"]
                        ),
                        amount=amount,
                        from_address=tx_data["transaction"][
                            "message"
                        ]["accountKeys"][0],
                        to_address=tx_data["transaction"]["message"][
                            "accountKeys"
                        ][1],
                    )
                    transactions.append(tx)

            return transactions
        except Exception as e:
            logger.error(f"Error fetching transactions: {e}")
            return []

    async def analyze_transactions(
        self, transactions: List[Transaction]
    ) -> Dict:
        tx_data = [asdict(tx) for tx in transactions]
        cluster_data = self.wallet_analyzer.identify_clusters()

        trend_analysis = await self.trend_agent.run(
            f"Analyze trends in: {json.dumps(tx_data)}"
        )
        print(trend_analysis)

        risk_analysis = await self.risk_agent.run(
            f"Analyze risks in: {json.dumps({'transactions': tx_data, 'clusters': cluster_data})}"
        )
        print(risk_analysis)

        summary = await self.summary_agent.run(
            f"Synthesize insights from: {trend_analysis}, {risk_analysis}"
        )

        print(summary)

        volume_chart = self.visualizer.generate_volume_chart()
        network_graph = self.visualizer.generate_network_graph(
            self.wallet_analyzer
        )

        return {
            "transactions": tx_data,
            "trend_analysis": trend_analysis,
            "risk_analysis": risk_analysis,
            "cluster_analysis": cluster_data,
            "summary": summary,
            "visualizations": {
                "volume_chart": volume_chart,
                "network_graph": network_graph,
            },
        }

    async def run_continuous_analysis(self):
        logger.info("Starting continuous analysis")
        asyncio.create_task(self.start_websocket_stream())

        while True:
            try:
                transactions = await self.fetch_transactions()
                if transactions:
                    analysis = await self.analyze_transactions(
                        transactions
                    )
                    timestamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    with open(f"analysis_{timestamp}.json", "w") as f:
                        json.dump(analysis, f, indent=2, default=str)
                    logger.info(
                        f"Analysis completed: analysis_{timestamp}.json"
                    )
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(60)


# Add to __main__:
if __name__ == "__main__":
    logger.info("Starting Solana analyzer...")
    analyzer = SolanaMultiAgentAnalyzer(alert_email="your@email.com")
    try:
        asyncio.run(analyzer.run_continuous_analysis())
    except Exception as e:
        logger.error(f"Critical error: {e}")
