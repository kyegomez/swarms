from swarms.agents.reasoning_duo import ReasoningDuo

reasoning_duo = ReasoningDuo(
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
    model_names=["gpt-4o-mini", "gpt-4o-mini"],
)

reasoning_duo.run(
    "What is the best possible financial strategy to maximize returns but minimize risk? Give a list of etfs to invest in and the percentage of the portfolio to allocate to each etf."
)

reasoning_duo.batched_run(
    [
        "What is the best possible financial strategy to maximize returns but minimize risk? Give a list of etfs to invest in and the percentage of the portfolio to allocate to each etf.",
        "What is the best possible financial strategy to maximize returns but minimize risk? Give a list of etfs to invest in and the percentage of the portfolio to allocate to each etf.",
    ]
)
