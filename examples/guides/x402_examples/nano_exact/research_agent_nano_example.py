import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from nano_exact import (
    MATCHED,
    RAW_PER_XNO,
    NanoExactSeller,
    NanoNode,
)
from swarms import Agent
from swarms_tools import exa_search

load_dotenv()

pay_to = os.getenv("NANO_PAY_TO")
node_url = os.getenv("NANO_NODE_URL")
if not pay_to or not node_url:
    raise SystemExit("Set NANO_PAY_TO and NANO_NODE_URL to run this.")

seller = NanoExactSeller(
    pay_to=pay_to,
    price_raw=int(os.getenv("NANO_PRICE_RAW", RAW_PER_XNO // 1000)),
    node=NanoNode(node_url),
    state_dir=os.getenv("NANO_STATE_DIR", ".nano_state"),
)

app = FastAPI(title="Research Agent API")

research_agent = Agent(
    agent_name="Research-Agent",
    system_prompt="You are an expert research analyst. Conduct thorough research on the given topic and provide comprehensive, well-structured insights with citations.",
    model_name="gpt-5.4",
    max_loops=1,
    tools=[exa_search],
)


@app.get("/research")
async def conduct_research(request: Request, query: str):
    """
    Run the research agent once for each paid nano:mainnet order.

    Args:
        request: the incoming request, read for its X-PAYMENT header
        query: The research topic or question

    Returns:
        Research results, or a 402 offer bound to this request
    """
    resource = request.url.path + "?" + request.url.query
    header = request.headers.get("X-PAYMENT")
    if header:
        verdict = seller.verify(header, resource)
        if verdict == MATCHED:
            return {"research": research_agent.run(query)}
        body = seller.offer(resource, "Research agent")
        body["error"] = verdict
        return JSONResponse(body, status_code=402)
    return JSONResponse(
        seller.offer(resource, "Research agent"), status_code=402
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
