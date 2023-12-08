import requests
from ..tool import Tool
from typing import Any
import os
import xmltodict


def build_tool(config) -> Tool:
    tool = Tool(
        "Wolfram",
        "Wolfram",
        name_for_model="Wolfram",
        name_for_human="Wolfram",
        description_for_model=""""Dynamic computation and curated data from WolframAlpha and Wolfram Cloud.\nOnly use the getWolframAlphaResults endpoints; all other Wolfram endpoints are deprecated.\nPrefer getWolframAlphaResults unless Wolfram Language code should be evaluated.\nTry to include images returned by getWolframAlphaResults. Queries to getWolframAlphaResults must ALWAYS have this structure: {\"input\": query}.\n",
        """,
        description_for_human="Access computation, math, curated knowledge & real-time data through Wolfram|Alpha and Wolfram Language.",
        logo_url="https://www.wolframcdn.com/images/icons/Wolfram.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    @tool.get("/getWolframAlphaResults")
    def getWolframAlphaResults(input: str):
        """Get Wolfram|Alpha results using natural query. Queries to getWolframAlphaResults must ALWAYS have this structure: {\"input\": query}. And please directly read the output json."""
        URL = "https://api.wolframalpha.com/v2/query"

        APPID = config["subscription_key"]

        params = {"appid": APPID, "input": input}

        response = requests.get(URL, params=params)

        json_data = xmltodict.parse(response.text)

        if "pod" not in json_data["queryresult"]:
            return "WolframAlpha API cannot parse the input query."
        rets = json_data["queryresult"]["pod"]

        cleaned_rets = []
        blacklist = [
            "@scanner",
            "@id",
            "@position",
            "@error",
            "@numsubpods",
            "@width",
            "@height",
            "@type",
            "@themes",
            "@colorinvertable",
            "expressiontypes",
        ]

        def filter_dict(d, blacklist):
            if isinstance(d, dict):
                return {
                    k: filter_dict(v, blacklist)
                    for k, v in d.items()
                    if k not in blacklist
                }
            elif isinstance(d, list):
                return [filter_dict(i, blacklist) for i in d]
            else:
                return d

        for ret in rets:
            ret = filter_dict(ret, blacklist=blacklist)
            # Do further cleaning to retain only the input and result pods
            if "@title" in ret:
                if ret["@title"] == "Input" or ret["@title"] == "Result":
                    cleaned_rets.append(ret)

        return cleaned_rets

    return tool
