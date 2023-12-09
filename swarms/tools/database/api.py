#!/usr/bin/env python
# coding=utf-8

import json
import os

from ..tool import Tool
from swarms.tools.database.utils.db_parser import get_conf
from swarms.tools.database.utils.database import DBArgs, Database

import openai

from typing import Optional, List, Mapping, Any
import requests, json


def build_database_tool(config) -> Tool:
    tool = Tool(
        "Data in a database",
        "Look up user data",
        name_for_model="Database",
        description_for_model="Plugin for querying the data in a database",
        logo_url="https://commons.wikimedia.org/wiki/File:Postgresql_elephant.svg",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    URL_REWRITE = "http://8.131.229.55:5114/rewrite"

    # load db settings
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    config = get_conf(script_dir + "/my_config.ini", "postgresql")
    dbargs = DBArgs("postgresql", config=config)  # todo assign database name

    # send request to database
    db = Database(dbargs, timeout=-1)
    schema = ""
    query = ""

    @tool.get("/get_database_schema")
    # def get_database_schema(query : str='select * from customer limit 2;', db_name : str='tpch10x'):
    def get_database_schema(db_name: str = "tpch10x"):
        global schema

        # todo simplify the schema based on the query
        print("=========== database name:", db_name)
        schema = db.compute_table_schema()

        print("========schema:", schema)

        text_output = f"The database schema is:\n" + "".join(str(schema))

        return text_output

    @tool.get("/translate_nlp_to_sql")
    def translate_nlp_to_sql(description: str):
        global schema, query
        """translate_nlp_to_sql(description: str) translates the input nlp string into sql query based on the database schema, and the sql query is the input of rewrite_sql and select_database_data API.
            description is a string that represents the description of the result data.
            schema is a string that represents the database schema.
            Final answer should be complete.

            This is an example:
            Thoughts: Now that I have the database schema, I will use the \\\'translate_nlp_to_sql\\\' command to generate the SQL query based on the given description and schema, and take the SQL query as the input of the \\\'rewrite_sql\\\' and  \\\'select_database_data\\\' commands.
            Reasoning: I need to generate the SQL query accurately based on the given description. I will use the \\\'translate_nlp_to_sql\\\' command to obtain the SQL query based on the given description and schema, and take the SQL query as the input of the \\\'select_database_data\\\' command.
            Plan: - Use the \\\'translate_nlp_to_sql\\\' command to generate the SQL query. \\\\n- Use the \\\'finish\\\' command to signal that I have completed all my objectives.
            Command: {"name": "translate_nlp_to_sql", "args": {"description": "Retrieve the comments of suppliers . The results should be sorted in descending order based on the comments of the suppliers."}}
            Result: Command translate_nlp_to_sql returned: "SELECT s_comment FROM supplier BY s_comment DESC"
        """

        openai.api_key = os.environ["OPENAI_API_KEY"]
        # schema = db.compute_table_schema()

        prompt = """Translate the natural language description into an semantic equivalent SQL query.
        The table and column names used in the sql must exactly appear in the schema. Any other table and column names are unacceptable.
        The schema is:\n
        {}
        
        The description is:\n
        {}

        The SQL query is:
        """.format(
            schema, description
        )

        # Set up the OpenAI GPT-3 model
        model_engine = "gpt-3.5-turbo"

        prompt_response = openai.ChatCompletion.create(
            engine=model_engine,
            messages=[
                {
                    "role": "assistant",
                    "content": "The table schema is as follows: " + schema,
                },
                {"role": "user", "content": prompt},
            ],
        )
        output_text = prompt_response["choices"][0]["message"]["content"]

        query = output_text

        return output_text

    @tool.get("/select_database_data")
    def select_database_data(query: str):
        """select_database_data(query : str) Read the data stored in database based on the SQL query from the translate_nlp_to_sql API.
        query : str is a string that represents the SQL query outputted by the translate_nlp_to_sql API.
        Final answer should be complete.

        This is an example:
        Thoughts: Now that I have the database schema and SQL query, I will use the \\\'select_database_data\\\' command to retrieve the data from the database based on the SQL query
        Reasoning: I will use the \\\'select_database_data\\\' command to retrieve the data from the database based on the SQL query
        Plan: - Use the \\\'select_database_data\\\' command to retrieve the data from the database based on the SQL query.\\\\n- Use the \\\'finish\\\' command to signal that I have completed all my objectives.
        Command: {"name": "select_database_data", "args": {query: "SELECT s_comment FROM supplier BY s_comment DESC"}}
        Result: Command select_database_data returned: "The number of result rows is: 394"
        """

        if query == "":
            raise RuntimeError("SQL query is empty")

        print("=========== database query:", query)
        res_completion = db.pgsql_results(query)  # list format

        if res_completion == "<fail>":
            raise RuntimeError("Database query failed")

        # data = json.loads(str(res_completion).strip())
        if isinstance(res_completion, list):
            text_output = f"The number of result rows is: " + "".join(
                str(len(res_completion))
            )
        else:
            text_output = f"The number of result rows is: " + "".join(
                str(res_completion)
            )

        return text_output

    @tool.get("/rewrite_sql")
    def rewrite_sql(
        sql: str = "select distinct l_orderkey, sum(l_extendedprice + 3 + (1 - l_discount)) as revenue, o_orderkey, o_shippriority from customer, orders, lineitem where c_mktsegment = 'BUILDING' and c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate < date '1995-03-15' and l_shipdate > date '1995-03-15' group by l_orderkey, o_orderkey, o_shippriority order by revenue desc, o_orderkey;",
    ):
        """Rewrite the input sql query"""

        param = {"sql": sql}
        print("Rewriter param:", param)
        headers = {"Content-Type": "application/json"}
        res_completion = requests.post(
            URL_REWRITE, data=json.dumps(param), headers=headers
        )

        # print("============ res_completion", res_completion.text)

        data = json.loads(res_completion.text.strip())
        data = data.get("data")
        text_output = f"Rewritten sql is:\n" + data.get("rewritten_sql")

        return text_output

    return tool
