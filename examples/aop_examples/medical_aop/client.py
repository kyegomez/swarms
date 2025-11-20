import asyncio
from typing import Dict

import requests

from swarms.structs.aop import AOPCluster
from swarms.tools.mcp_client_tools import execute_tool_call_simple


def _select_tools_by_keyword(tools: list, keyword: str) -> list:
    """
    Return tools whose name or description contains the keyword
    (case-insensitive).
    """
    kw = keyword.lower()
    selected = []
    for t in tools:
        name = t.get("function", {}).get("name", "")
        desc = t.get("function", {}).get("description", "")
        if kw in name.lower() or kw in desc.lower():
            selected.append(t)
    return selected


def _example_payload_from_schema(tools: list, tool_name: str) -> dict:
    """
    Construct a minimal example payload for a given tool using its JSON schema.
    Falls back to a generic 'task' if schema not present.
    """
    for t in tools:
        fn = t.get("function", {})
        if fn.get("name") == tool_name:
            schema = fn.get("parameters", {})
            required = schema.get("required", [])
            props = schema.get("properties", {})
            payload = {}
            for r in required:
                if r in props:
                    if props[r].get("type") == "string":
                        payload[r] = (
                            "Example patient case: 45M, egfr 59 ml/min/1.73"
                        )
                    elif props[r].get("type") == "boolean":
                        payload[r] = False
                    else:
                        payload[r] = None
            if not payload:
                payload = {
                    "task": "Provide ICD-10 suggestions for the case above"
                }
            return payload
    return {"task": "Provide ICD-10 suggestions for the case above"}


def main() -> None:
    cluster = AOPCluster(
        urls=["http://localhost:8000/mcp"],
        transport="streamable-http",
    )

    tools = cluster.get_tools(output_type="dict")
    print(f"Tools: {len(tools)}")

    coding_tools = _select_tools_by_keyword(tools, "coder")
    names = [t.get("function", {}).get("name") for t in coding_tools]
    print(f"Coding-related tools: {names}")

    # Build a real payload for "Medical Coder" and execute the tool call
    tool_name = "Medical Coder"
    payload: Dict[str, object] = _example_payload_from_schema(
        tools, tool_name
    )

    # Enrich with public keyless data (epidemiology context via disease.sh)
    try:
        epi = requests.get(
            "https://disease.sh/v3/covid-19/countries/USA?strict=true",
            timeout=5,
        )
        if epi.ok:
            data = epi.json()
            epi_summary = (
                f"US COVID-19 context: cases={data.get('cases')}, "
                f"todayCases={data.get('todayCases')}, deaths={data.get('deaths')}"
            )
            base_task = payload.get("task") or ""
            payload["task"] = (
                f"{base_task}\n\nEpidemiology context (no key API): {epi_summary}"
            )
    except Exception:
        pass

    print("Calling tool:", tool_name)
    request = {
        "function": {
            "name": tool_name,
            "arguments": payload,
        }
    }
    result = asyncio.run(
        execute_tool_call_simple(
            response=request,
            server_path="http://localhost:8000/mcp",
            output_type="json",
            transport="streamable-http",
            verbose=False,
        )
    )
    print("Response:")
    print(result)


if __name__ == "__main__":
    main()
