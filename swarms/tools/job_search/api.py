import requests
import json
from datetime import date, datetime, timedelta
import os
from ..tool import Tool

from typing import Optional, Dict, Union, List


def build_tool(config) -> Tool:
    tool = Tool(
        "Search job information",
        "Search job information in Linkin, Glassdoor, etc.",
        name_for_model="JobSearch",
        description_for_model="Plugin for look up job information in Linkin, Glassdoor, etc.",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    BASE_URL = "https://jsearch.p.rapidapi.com"
    KEY = config["subscription_key"]
    HEADERS = {"X-RapidAPI-Key": KEY, "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"}

    @tool.get("/basic_job_search")
    def basic_job_search(
        query: str,
        page: int = 1,
        num_pages: str = "1",
        date_posted: Optional[str] = None,
        remote_jobs_only: Optional[bool] = None,
        employment_types: Optional[str] = None,
        job_requirements: Optional[str] = None,
        job_titles: Optional[str] = None,
        company_types: Optional[str] = None,
        employer: Optional[str] = None,
        radius: Optional[int] = None,
        categories: Optional[str] = None,
    ) -> dict:
        """
        Search for jobs posted on job sites across the web.

        :param query: Free-form jobs search query.
        :param page: Page to return (each page includes up to 10 results).
        :param num_pages: Number of pages to return, starting from page.
        :param date_posted: Find jobs posted within the time you specify.
        :param remote_jobs_only: Find remote jobs only (work from home).
        :param employment_types: Find jobs of particular employment types.
        :param job_requirements: Find jobs with specific requirements.
        :param job_titles: Find jobs with specific job titles.
        :param company_types: Find jobs posted by companies of certain types.
        :param employer: Find jobs posted by specific employers.
        :param radius: Return jobs within a certain distance from location as specified as part of the query (in km).
        :param categories: [Deprecated] Find jobs in specific categories/industries.
        :return: A dictionary with the response from the API.
        """

        querystring = {"query": query, "page": page, "num_pages": num_pages}

        if date_posted:
            querystring["date_posted"] = date_posted
        if remote_jobs_only is not None:
            querystring["remote_jobs_only"] = remote_jobs_only
        if employment_types:
            querystring["employment_types"] = employment_types
        if job_requirements:
            querystring["job_requirements"] = job_requirements
        if job_titles:
            querystring["job_titles"] = job_titles
        if company_types:
            querystring["company_types"] = company_types
        if employer:
            querystring["employer"] = employer
        if radius:
            querystring["radius"] = radius
        if categories:
            querystring["categories"] = categories

        response = requests.get(
            BASE_URL + "/search", headers=HEADERS, params=querystring
        )

        return response.json()

    @tool.get("/search_jobs_with_filters")
    def search_jobs_with_filters(
        self,
        query: str,
        page: int = 1,
        num_pages: int = 1,
        date_posted: str = "all",
        remote_jobs_only: bool = False,
        employment_types: Optional[str] = None,
        job_requirements: Optional[str] = None,
        job_titles: Optional[str] = None,
        company_types: Optional[str] = None,
        employer: Optional[str] = None,
        radius: Optional[int] = None,
        categories: Optional[str] = None,
    ) -> dict:
        """
        Search for jobs using the JSearch API.

        Args:
            query (str): The job search query.Query examples,web development in chicago, marketing manager in new york via linkedin,developer in germany 60306
            page (int, optional): The page to return. Defaults to 1.
            num_pages (int, optional): The number of pages to return. Defaults to 1.
            date_posted (str, optional): Find jobs posted within the time you specify. Defaults to "all". Find jobs posted within the time you specify.Possible values: all, today, 3days, week,month.Default: all.
            remote_jobs_only (bool, optional): Find remote jobs only. Defaults to False.
            employment_types (str, optional): Find jobs of particular employment types.
            job_requirements (str, optional): Find jobs with specific requirements.
            job_titles (str, optional): Find jobs with specific job titles.
            company_types (str, optional): Find jobs posted by companies of certain types.
            employer (str, optional): Find jobs posted by specific employers.
            radius (int, optional): Return jobs within a certain distance from location.
            categories (str, optional): Find jobs in specific categories/industries.

        Returns:
            dict: The JSON response from the API.
        """

        params = {
            "query": query,
            "page": str(page),
            "num_pages": str(num_pages),
            "date_posted": date_posted,
            "remote_jobs_only": str(remote_jobs_only).lower(),
            "employment_types": employment_types,
            "job_requirements": job_requirements,
            "job_titles": job_titles,
            "company_types": company_types,
            "employer": employer,
            "radius": str(radius) if radius else None,
            "categories": categories,
        }

        # remove None values in the parameters
        params = {k: v for k, v in params.items() if v is not None}

        response = requests.get(
            BASE_URL + "/search-filters", headers=HEADERS, params=params
        )

        return response.json()

    @tool.get("/get_job_details")
    def get_job_details(
        job_ids: Union[str, List[str]], extended_publisher_details: bool = False
    ) -> dict:
        """
        You can get the job_ids from 'basic_job_search' function
        Get all job details, including additional application options / links, employer reviews and estimated salaries for similar jobs.

        :param job_ids: Job Id of the job for which to get details. Batching of up to 20 Job Ids is supported by separating multiple Job Ids by comma (,).
                        Note that each Job Id in a batch request is counted as a request for quota calculation.
        :param extended_publisher_details: [BETA] Return additional publisher details such as website url and favicon.
        :return: A dictionary with the response from the API.
        """

        if isinstance(job_ids, list):
            job_ids = ",".join(job_ids)

        querystring = {
            "job_id": job_ids,
            "extended_publisher_details": str(extended_publisher_details).lower(),
        }

        response = requests.get(
            BASE_URL + "/job-details", headers=HEADERS, params=querystring
        )

        return response.json()

    @tool.get("/get_salary_estimation")
    def get_salary_estimation(
        job_title: Optional[str] = None,
        location: Optional[str] = None,
        radius: int = 200,
    ) -> dict:
        """
        Get estimated salaries for a jobs around a location.

        :param job_title: Job title for which to get salary estimation.
        :param location: Location in which to get salary estimation.
        :param radius: Search radius in km (measured from location).
        :return: A dictionary with the response from the API.
        """

        querystring = {}

        if job_title:
            querystring["job_title"] = job_title
        if location:
            querystring["location"] = location
        if radius:
            querystring["radius"] = radius

        response = requests.get(
            BASE_URL + "/estimated-salary", headers=HEADERS, params=querystring
        )

        return response.json()

    return tool
