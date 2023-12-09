import requests
import json
from datetime import date, datetime, timedelta
import os
from ..tool import Tool
from typing import Optional, List, Dict, Any
from serpapi import GoogleSearch


def build_tool(config) -> Tool:
    tool = Tool(
        "Google Scholar Info",
        "Look up google scholar information",
        name_for_model="Google_Scholar",
        description_for_model="Plugin for look up Google Scholar information",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    KEY = config["subscription_key"]

    @tool.get("/search_google_scholar")
    def search_google_scholar(
        query: str,
        engine: str = "google_scholar",
        cites: Optional[str] = None,
        as_ylo: Optional[int] = None,
        as_yhi: Optional[int] = None,
        scisbd: Optional[int] = None,
        cluster: Optional[str] = None,
        hl: Optional[str] = None,
        lr: Optional[str] = None,
        start: Optional[int] = None,
        num: Optional[int] = None,
        as_sdt: Optional[str] = None,
        safe: Optional[str] = None,
        filter: Optional[str] = None,
        as_vis: Optional[str] = None,
    ):
        """
        Search for scholarly articles based on a query according to the google scholar
        :param query: The query to search for.
        :param engine: The search engine to use, default is "google_scholar"
        :param cites: The unique ID of an article for triggering "Cited By" searches
        :param as_ylo: The starting year for results (e.g., if as_ylo=2018, results before this year will be omitted)
        :param as_yhi: The ending year for results (e.g., if as_yhi=2018, results after this year will be omitted)
        :param scisbd: Defines articles added in the last year, sorted by date. It can be set to 1 to include only abstracts, or 2 to include everything
        :param cluster: The unique ID of an article for triggering "All Versions" searches
        :param hl: The language to use for the Google Scholar search
        :param lr: One or multiple languages to limit the search to
        :param start: The result offset for pagination (0 is the first page of results, 10 is the 2nd page, etc.)
        :param num: The maximum number of results to return, limited to 20
        :param as_sdt: Can be used either as a search type or a filter
        :param safe: The level of filtering for adult content
        :param filter: Defines if the filters for 'Similar Results' and 'Omitted Results' are on or off
        :param as_vis: Defines whether to include citations or not
        :return: Return a list of dictionaries of the papers
        """
        params = {
            "q": query,
            "engine": engine,
            "api_key": KEY,
            "cites": cites,
            "as_ylo": as_ylo,
            "as_yhi": as_yhi,
            "scisbd": scisbd,
            "cluster": cluster,
            "hl": hl,
            "lr": lr,
            "start": start,
            "num": num,
            "as_sdt": as_sdt,
            "safe": safe,
            "filter": filter,
            "as_vis": as_vis,
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results["organic_results"]

        return organic_results

    @tool.get("/search_author")
    def search_author(
        author_id: str,
        hl: Optional[str] = None,
        view_op: Optional[str] = None,
        sort: Optional[str] = None,
        citation_id: Optional[str] = None,
        start: Optional[int] = None,
        num: Optional[int] = None,
        no_cache: Optional[bool] = None,
        async_req: Optional[bool] = None,
        output: Optional[str] = None,
    ):
        """
        Search for an author using the Google Scholar Author API.
        :param author_id: Required. The ID of an author.
        :param hl: Optional. The language to use for the Google Scholar Author search. Default is 'en'.
        :param view_op: Optional. Used for viewing specific parts of a page.
        :param sort: Optional. Used for sorting and refining articles.
        :param citation_id: Optional. Used for retrieving individual article citation.
        :param start: Optional. Defines the result offset. Default is 0.
        :param num: Optional. Defines the number of results to return. Default is 20.
        :param no_cache: Optional. Forces SerpApi to fetch the results even if a cached version is already present. Default is False.
        :param async_req: Optional. Defines the way you want to submit your search to SerpApi. Default is False.
        :param output: Optional. Defines the final output you want. Default is 'json'.
        :return: Returns the search results of the author basic information.
        """
        params = {
            "engine": "google_scholar_author",
            "author_id": author_id,
            "api_key": KEY,
            "hl": hl,
            "view_op": view_op,
            "sort": sort,
            "citation_id": citation_id,
            "start": start,
            "num": num,
            "no_cache": no_cache,
            "async": async_req,
            "output": output,
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        author = results["author"]

        return author

    @tool.get("/get_citation")
    def get_citation(
        q: str,
        no_cache: Optional[bool] = None,
        async_: Optional[bool] = None,
        output: Optional[str] = "json",
    ) -> Dict[str, Any]:
        """
        Function to get citation results from the Google Scholar organic results using the Google Scholar Cite API.

        Parameters:
        q (str): ID of an individual Google Scholar organic search result.
        engine (str, optional): Set to 'google_scholar_cite' to use the Google Scholar API engine. Defaults to 'google_scholar_cite'.
        no_cache (bool, optional): If set to True, will force SerpApi to fetch the Google Scholar Cite results even if a cached version is already present. Defaults to None.
        async_ (bool, optional): If set to True, will submit search to SerpApi and retrieve results later. Defaults to None.
        api_key (str): SerpApi private key to use.
        output (str, optional): Final output format. Set to 'json' to get a structured JSON of the results, or 'html' to get the raw html retrieved. Defaults to 'json'.

        Returns:
        Dict[str, Any]: Returns the search results in the specified format.
        """

        params = {
            "q": q,
            "engine": "google_scholar_cite",
            "no_cache": no_cache,
            "async": async_,
            "api_key": KEY,
            "output": output,
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        citation = results["citations"]

        return citation

    @tool.get("/get_profile")
    def get_profile(
        self,
        mauthors: str,
        hl: Optional[str] = "en",
        after_author: Optional[str] = None,
        before_author: Optional[str] = None,
        no_cache: Optional[bool] = False,
        _async: Optional[bool] = False,
        output: Optional[str] = "json",
    ) -> Dict:
        """
        The getProfile function is used to scrape profile results from the Google Scholar Profiles search page.

        Args:
            mauthors (str): Defines the author you want to search for.
            hl (str, optional): Defines the language to use for the Google Scholar Profiles search.
                It's a two-letter language code. (e.g., 'en' for English, 'es' for Spanish, or 'fr' for French).
                Defaults to 'en'.
            after_author (str, optional): Defines the next page token.
                It is used for retrieving the next page results.
                The parameter has the precedence over before_author parameter. Defaults to None.
            before_author (str, optional): Defines the previous page token.
                It is used for retrieving the previous page results. Defaults to None.
            no_cache (bool, optional): Will force SerpApi to fetch the Google Scholar Profiles results even if a cached version is already present.
                Defaults to False.
            _async (bool, optional): Defines the way you want to submit your search to SerpApi. Defaults to False.
            api_key (str): Defines the SerpApi private key to use.
            output (str, optional): Defines the final output you want.
                It can be set to 'json' (default) to get a structured JSON of the results,
                or 'html' to get the raw html retrieved. Defaults to 'json'.

        Returns:
            Dict: The Google Scholar Profiles search results.
        """
        params = {
            "mauthors": mauthors,
            "engine": "google_scholar_profiles",
            "hl": hl,
            "after_author": after_author,
            "before_author": before_author,
            "engine": "google_scholar_profiles",
            "no_cache": no_cache,
            "async": _async,
            "api_key": KEY,
            "output": output,
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        profile = results["profiles"]

        return profile

    return tool
