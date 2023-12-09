from .utils import *
import pandas as pd
import requests
import json
from ...tool import Tool


def build_tool(config) -> Tool:
    tool = Tool(
        "Search in Wikidata",
        "answering factual questions in wikidata.",
        description_for_model="Plugin for answering factual questions in wikidata.",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )
    sparql = Slot2Sparql()

    @tool.get("/find_entity")
    def find_entity(input):
        """Find all <r, t> that has the relation <input, r, t>. It looks like viewing the main page of the input entity. The result is a table."""
        try:
            sparqlIdx = -1

            if input[0] == "#":
                input = {"id": int(input[1:]), "attr": "tmp"}
            elif input[0] == "Q":
                input = {"id": input, "attr": "wd"}
            elif input[0] == "P":
                input = {"id": input, "attr": "wdt"}
            elif input[0] == "@":
                input = {"id": input[1:], "attr": "wds"}
            else:
                input = {"id": input, "attr": "val"}
            sparql.find_entity(input)
            sparqlIdx = len(sparql.select_lst) - 1

            query, ids = sparql.give_str(sparqlIdx)

            query += "\nLIMIT 2000"

            ids = ["#" + str(id["id"]) for id in ids]

            result = getResult(query)

            variable_name = [
                enc(sparql.select_lst[sparqlIdx].state[-1][1])[1:],
                enc(sparql.select_lst[sparqlIdx].state[-1][2])[1:],
                enc(sparql.select_lst[sparqlIdx].state[-1][3])[1:],
            ]

            response = [{} for i in range(0, len(result))]

            print("RESULT:", result)

            for idx, re in enumerate(result):
                response[idx].update(
                    get_property_details(re[variable_name[0]]["value"])
                    if re[variable_name[0]]["type"] == "uri"
                    else {
                        "relation": "",
                        "relationLabel": re[variable_name[0]]["value"],
                        "relationDescription": "",
                        # 'propuri': ''
                    }
                )

                response[idx].update(
                    {
                        "tail": re[variable_name[1]]["value"].split("/")[-1]
                        if re[variable_name[1]]["type"] == "uri"
                        else "",
                        "tailLabel": re.get(variable_name[1] + "Label", {"value": ""})[
                            "value"
                        ]
                        if re[variable_name[1]]["type"] == "uri"
                        else re[variable_name[1]]["value"],
                        "tailDescription": re.get(
                            variable_name[1] + "Description", {"value": ""}
                        )["value"],
                        # 'tailuri': re[variable_name[1]]['value'] if re[variable_name[1]]['type'] == 'uri' else '',
                        # 'tailtype': 'uri' if re[variable_name[1]]['type'] == 'uri' else re[variable_name[1]].get('datatype', '')
                    }
                )
                if variable_name[2] in re:
                    response[idx].update(
                        {
                            "time": re.get(variable_name[2] + "Label", {"value": ""})[
                                "value"
                            ]
                            if re[variable_name[2]]["type"] == "uri"
                            else re[variable_name[2]]["value"],
                        }
                    )
                else:
                    response[idx].update({"time": "ALWAYS"})

            df = pd.DataFrame.from_dict(response)
            return df.to_markdown()
        except Exception:
            print("Invalid option!\n", Exception)
            return df.to_markdown()

    @tool.get("/find_entity_by_tail")
    def find_entity_by_tail(input: str):
        """Find all <h, r> that has the relation <h, r, input>. It looks like viewing the reverse main page of the input entity. The result is a table."""
        try:
            sparqlIdx = -1

            if input[0] == "#":
                input = {"id": int(input[1:]), "attr": "tmp"}
            elif input[0] == "Q":
                input = {"id": input, "attr": "wd"}
            elif input[0] == "P":
                input = {"id": input, "attr": "wdt"}
            elif input[0] == "@":
                input = {"id": input[1:], "attr": "wds"}
            else:
                input = {"id": input, "attr": "val"}
            sparql.find_entity_by_tail(input)
            sparqlIdx = len(sparql.select_lst) - 1

            query, ids = sparql.give_str(sparqlIdx)

            query += "\nLIMIT 2000"

            ids = ["#" + str(id["id"]) for id in ids]

            result = getResult(query)

            variable_name = [
                enc(sparql.select_lst[sparqlIdx].state[-1][0])[1:],
                enc(sparql.select_lst[sparqlIdx].state[-1][1])[1:],
            ]

            response = [{} for i in range(0, len(result))]

            for idx, re in enumerate(result):
                response[idx].update(
                    get_property_details(re[variable_name[1]]["value"])
                    if re[variable_name[1]]["type"] == "uri"
                    else {
                        "relation": "",
                        "relationLabel": re[variable_name[1]]["value"],
                        "relationDescription": "",
                        # 'labelUri': ''
                    }
                )

                response[idx].update(
                    {
                        "head": re[variable_name[0]]["value"].split("/")[-1]
                        if re[variable_name[0]]["type"] == "uri"
                        else "",
                        "headLabel": re.get(variable_name[0] + "Label", {"value": ""})[
                            "value"
                        ]
                        if re[variable_name[0]]["type"] == "uri"
                        else re[variable_name[0]]["value"],
                        "headDescription": re.get(
                            variable_name[0] + "Description", {"value": ""}
                        )["value"],
                        # 'headUri': re[variable_name[0]]['value'] if re[variable_name[0]]['type'] == 'uri' else '',
                        # 'headType': 'uri' if re[variable_name[0]]['type'] == 'uri' else re[variable_name[0]].get('datatype', '')
                    }
                )

            df = pd.DataFrame.from_dict(response)
            return df.to_markdown()

        except Exception:
            print("Invalid option!\n", Exception)
            return pd.DataFrame().to_markdown()

    @tool.get("/get_entity_id")
    def get_entity_id(input: str):
        """Search for all the entities that has the surface form as the input. For example, all the entities that are named ``Obama'', including either person, book, anything else."""
        try:
            result = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "type": "item",
                    "action": "wbsearchentities",
                    "language": "en",
                    "search": input,
                    "origin": "*",
                    "format": "json",
                },
            ).text

            result = json.loads(result)["search"]
            # print(result)

            df = pd.DataFrame.from_dict(result)
            for row in df.axes[1]:
                if row != "id" and row != "label" and row != "description":
                    df.pop(row)
            return df.to_markdown()

        except Exception:
            print("Invalid option!\n", Exception)
            return pd.DataFrame().to_markdown()

    @tool.get("/get_relation_id")
    def get_relation_id(input: str):
        """Search for all the relations that has the surface form as the input. For example, all the relations that are named ``tax''."""
        try:
            result = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "type": "property",
                    "action": "wbsearchentities",
                    "language": "en",
                    "search": input,
                    "origin": "*",
                    "format": "json",
                },
            ).text

            result = json.loads(result)["search"]

            df = pd.DataFrame.from_dict(result)
            for row in df.axes[1]:
                if row != "id" and row != "label" and row != "description":
                    df.pop(row)
            return df.to_markdown()

        except Exception:
            print("Invalid option!\n", Exception)
            return pd.DataFrame().to_markdown()

    @tool.get("/search_by_code")
    def search_by_code(query: str):
        """After knowing the unique id of entity or relation, perform a sparql query. E.g.,
        Select ?music\nWhere {{\nwd:Q00 wdt:P00 ?music.\n}} The entity label will be automatically retrieved.
        """
        try:
            query, basic_sel = convert_sparql_to_backend(query)

            result = getResult(query)

            for i in range(0, len(result)):
                for sel in basic_sel:
                    if sel not in result[i]:
                        continue
                    if (
                        len(result[i][sel]["value"]) < 4
                        or result[i][sel]["value"][0:4] != "http"
                    ):
                        continue
                    id = result[i][sel]["value"].split("/")[-1]

                    if type(id) == str and len(id) > 0 and id[0] == "P":
                        result[i].update(
                            convert(
                                get_property_details_with_name(
                                    result[i][sel]["value"], sel
                                )
                            )
                        )

            df = pd.DataFrame.from_dict(result)
            return df.to_markdown()
        except Exception:
            print("Invalid option!\n", Exception)
            return pd.DataFrame().to_markdown()

    return tool
