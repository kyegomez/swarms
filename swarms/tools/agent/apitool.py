"""Interface for tools."""
from inspect import signature
from typing import Any, Awaitable, Callable, Optional, Union

from langchain.agents import Tool as LangChainTool
from langchain.tools.base import BaseTool
import requests
import json
import aiohttp
import http.client

http.client._MAXLINE = 655360

from swarms.utils import get_logger

logger = get_logger(__name__)


class Tool(LangChainTool):
    tool_logo_md: str = ""


class RequestTool(BaseTool):
    """Tool that takes in function or coroutine directly."""

    description: str = ""
    func: Callable[[str], str]
    afunc: Callable[[str], str]
    coroutine: Optional[Callable[[str], Awaitable[str]]] = None
    max_output_len = 4000
    tool_logo_md: str = ""

    def _run(self, tool_input: str) -> str:
        """Use the tool."""
        return self.func(tool_input)

    async def _arun(self, tool_input: str) -> str:
        """Use the tool asynchronously."""
        ret = await self.afunc(tool_input)
        return ret

    def convert_prompt(self, params):
        lines = "Your input should be a json (args json schema): {{"
        for p in params:
            logger.debug(p)
            optional = not p["required"]
            description = p.get("description", "")
            if len(description) > 0:
                description = "(" + description + ")"

            lines += '"{name}" : {type}{desc}, '.format(
                name=p["name"],
                type=p["schema"]["type"],
                optional=optional,
                desc=description,
            )

        lines += "}}"
        return lines

    def __init__(self, root_url, func_url, method, request_info, **kwargs):
        """Store the function, description, and tool_name in a class to store the information"""
        url = root_url + func_url

        def func(json_args):
            if isinstance(json_args, str):
                try:
                    json_args = json.loads(json_args)
                except:
                    return "Your input can not be parsed as json, please use thought."
                if "tool_input" in json_args:
                    json_args = json_args["tool_input"]

            # if it's post put patch, then we do json
            if method.lower() in ["post", "put", "patch"]:
                response = getattr(requests, method.lower())(url, json=json_args)
            else:
                # for other methods, we use get, and use json_args as query params
                response = requests.get(url, params=json_args)
            if response.status_code == 200:
                message = response.text
            else:
                message = f"Error code {response.status_code}. You can try (1) Change your input (2) Call another function. (If the same error code is produced more than 4 times, please use Thought: I can not use these APIs, so I will stop. Final Answer: No Answer, please check the APIs.)"

            message = message[: self.max_output_len]  # TODO: not rigorous, to improve
            return message

        def convert_openapi_to_params(request_body):
            if not request_body:
                return []
            params = []
            for content_type, content in request_body["content"].items():
                schema = content["schema"]
                properties = schema.get("properties", {})
                required = schema.get("required", [])
                for key, value in properties.items():
                    param = {
                        "name": key,
                        "schema": value,
                        "required": key in required,
                        "description": value.get("description", ""),
                    }
                    if (
                        content_type == "multipart/form-data"
                        and value.get("format") == "binary"
                    ):
                        param["type"] = "file"
                    elif content_type in [
                        "application/x-www-form-urlencoded",
                        "multipart/form-data",
                    ]:
                        param["type"] = "form"
                    else:
                        param["type"] = "json"
                    params.append(param)
            return params

        async def afunc(json_args):
            if isinstance(json_args, str):
                try:
                    json_args = json.loads(json_args)
                except:
                    return "Your input can not be parsed as json, please use thought."
                if "tool_input" in json_args:
                    json_args = json_args["tool_input"]

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=json_args) as response:
                    if response.status == 200:
                        message = await response.text()
                    else:
                        message = f"Error code {response.status_code}. You can try (1) Change your input (2) Call another function. (If the same error code is produced more than 4 times, please use Thought: I can not use these APIs, so I will  stop. Final Answer: No Answer, please check the APIs.)"

            message = message[: self.max_output_len]  # TODO: not rigorous, to improve
            return message

        tool_name = func_url.replace("/", ".").strip(".")
        str_doc = ""
        if "parameters" in request_info[method]:
            str_doc = self.convert_prompt(request_info[method]["parameters"])

        if "requestBody" in request_info[method]:
            str_doc = (
                str_doc
                + "\n"
                + self.convert_prompt(
                    convert_openapi_to_params(request_info[method]["requestBody"])
                )
            )

        # description = f"- {tool_name}:\n" + \
        #      request_info[method].get('summary', '').replace("{", "{{").replace("}", "}}") \
        description = (
            request_info[method]
            .get("description", "")
            .replace("{", "{{")
            .replace("}", "}}")
            + ". "
            + str_doc
            + f" The Action to trigger this API should be {tool_name} and the input parameters should be a json dict string. Pay attention to the type of parameters."
        )

        logger.info("API Name: {}".format(tool_name))
        logger.info("API Description: {}".format(description))

        super(RequestTool, self).__init__(
            name=tool_name, func=func, afunc=afunc, description=description, **kwargs
        )
