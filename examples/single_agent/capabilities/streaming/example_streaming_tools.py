import json

from swarms import Agent


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def get_stock_price(ticker: str) -> str:
    """Return a mock current price for a stock ticker."""
    prices = {
        "AAPL": 213.45,
        "MSFT": 415.20,
        "NVDA": 875.30,
        "CEG": 192.60,
    }
    price = prices.get(ticker.upper(), 100.00)
    return json.dumps(
        {"ticker": ticker.upper(), "price": price, "currency": "USD"}
    )


def get_market_news(topic: str) -> str:
    """Return mock market headlines for a topic."""
    headlines = {
        "energy": "Nuclear power demand surges as data-center operators sign long-term PPAs.",
        "semiconductors": "NVDA posts record revenue driven by AI accelerator shipments.",
    }
    text = headlines.get(
        topic.lower(), f"Markets steady; awaiting data on {topic}."
    )
    return json.dumps({"topic": topic, "headline": text})


def calculate_pe_ratio(price: float, eps: float) -> str:
    """Calculate price-to-earnings ratio."""
    if eps <= 0:
        return json.dumps({"error": "EPS must be positive"})
    return json.dumps(
        {
            "price": price,
            "eps": eps,
            "pe_ratio": round(price / eps, 2),
        }
    )


TASK = (
    "Analyse Constellation Energy (CEG): fetch the current price, "
    "check the latest energy market news, and compute the P/E ratio assuming "
    "EPS of $8.40. Give me a buy / hold / sell recommendation."
)


def make_agent() -> Agent:
    return Agent(
        agent_name="Streaming-Trading-Agent",
        agent_description="Quantitative trading analyst",
        system_prompt=(
            "You are a quantitative trading analyst. "
            "Use the available tools to fetch live data before forming your view."
        ),
        model_name="claude-sonnet-4-6",
        max_loops="auto",
        thinking_tokens=2048,
        tools=[get_stock_price, get_market_news, calculate_pe_ratio],
        top_p=None,
        streaming_on=True,
        reasoning_effort="high",
        temperature=1,
    )


# ---------------------------------------------------------------------------
# 1. Sync generator — run_stream()
# ---------------------------------------------------------------------------


def demo_sync():
    print("\n" + "=" * 60)
    print("SYNC STREAMING  (run_stream)")
    print("=" * 60 + "\n")

    agent = make_agent()

    out = agent.run(TASK)
    print(out)


# ---------------------------------------------------------------------------
# 2. Async generator — arun_stream()
# ---------------------------------------------------------------------------


async def demo_async():
    print("\n" + "=" * 60)
    print("ASYNC STREAMING  (arun_stream)")
    print("=" * 60 + "\n")

    agent = make_agent()

    async for token in agent.arun_stream(TASK):
        print(token, end="", flush=True)

    print("\n")


# ---------------------------------------------------------------------------
# 3. Original blocking call — run()  (for comparison)
# ---------------------------------------------------------------------------


def demo_blocking():
    print("\n" + "=" * 60)
    print("BLOCKING  (run)")
    print("=" * 60 + "\n")

    agent = make_agent()
    agent.print_on = True  # let the agent render its own panels

    result = agent.run(TASK)
    print(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mode = "async"

    # if mode == "async":
    #     asyncio.run(demo_async())
    # elif mode == "blocking":
    #     demo_blocking()
    # else:
    demo_sync()
