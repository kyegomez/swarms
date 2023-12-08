import os
import re
import json
import requests
from ..tool import Tool
from typing import Optional
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from amadeus import Client, ResponseError


def build_tool(config) -> Tool:
    tool = Tool(
        "walmart Info",
        "Query about information about retail commodity on Walmart Platform.",
        name_for_model="walmart",
        description_for_model="""This is a plugin for look up real walmart infomation. Results from this API are inaccessible for users. Please organize and re-present them.""",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    SERPAPI_KEY = config["subscription_key"]

    @tool.get("/ItemQuery")
    def ItemQuery(
        item: str,
        option_num: Optional[int] = 3,
    ):
        """
        This API gather retail information about queried items at walmart
        :param item: product name presented as string.
        :param option_num: the number of items presented for each queried item.
        :return: a dict walmart retail information about queried items.
        """

        try:
            params = {
                "engine": "walmart",
                "query": item,
                "api_key": SERPAPI_KEY,
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            organic_results = results["organic_results"]

            item_res = []
            for idx in range(min(option_num, len(results["organic_results"]))):
                item_res.append({})
                item_res[idx]["name"] = organic_results[idx]["title"]
                item_res[idx]["description"] = organic_results[idx]["description"]
                item_res[idx]["price"] = organic_results[idx]["primary_offer"][
                    "offer_price"
                ]
                item_res[idx]["price_unit"] = organic_results[idx]["price_per_unit"][
                    "unit"
                ]
                item_res[idx]["url"] = organic_results[idx]["product_page_url"]
                item_res[idx]["rating"] = organic_results[idx]["rating"]
                item_res[idx]["seller_name"] = organic_results[idx]["seller_name"]

            return json.dumps(item_res)

        except Exception:  # Handle response error exceptions
            return {"error": "Response error."}

    return tool
