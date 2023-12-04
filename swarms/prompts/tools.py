# Prompts
DYNAMIC_STOP_PROMPT = """

Now, when you 99% sure you have completed the task, you may follow the instructions below to escape the autonomous loop.

When you have finished the task from the Human, output a special token: <DONE>
This will enable you to leave the autonomous loop.
"""


# Make it able to handle multi input tools
DYNAMICAL_TOOL_USAGE = """
You have access to the following tools:
Output a JSON object with the following structure to use the tools

commands: {
    "tools": {
        tool1: "search_api",
        "params": {
            "query": "What is the weather in New York?",
            "description": "Get the weather in New York"
        }
        "tool2: "weather_api",
        "params": {
            "query": "What is the weather in Silicon Valley",
        }
        "tool3: "rapid_api",
        "params": {
            "query": "Use the rapid api to get the weather in Silicon Valley",
        }
    }
}

"""







########### FEW SHOT EXAMPLES ################
SCENARIOS = """
commands: {
    "tools": {
        tool1: "function",
        "params": {
            "input": "inputs",
            "tool1": "inputs"
        }
        "tool2: "tool_name",
        "params": {
            "parameter": "inputs",
            "tool1": "inputs"
        }
        "tool3: "tool_name",
        "params": {
            "tool1": "inputs",
            "tool1": "inputs"
        }
    }
}

"""


def tools_prompt_prep(
    tool_docs: str = None, tool_few_shot_examples: str = None
):
    """
    Tools prompt prep

    Args:
        docs (str, optional): _description_. Defaults to None.
        scenarios (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    PROMPT = f"""
    # Task
    You will be provided with a list of APIs. These APIs will have a
    description and a list of parameters and return types for each tool. Your
    task involves creating varied, complex, and detailed user scenarios
    that require to call API calls. You must select what api to call based on 
    the context of the task and the scenario.

    For instance, given the APIs: SearchHotels, BookHotel, CancelBooking,
    GetNFLNews. Given that GetNFLNews is explicitly provided, your scenario
    should articulate something akin to:

    "The user wants to see if the Broncos won their last game (GetNFLNews).
    They then want to see if that qualifies them for the playoffs and who
    they will be playing against (GetNFLNews). The Broncos did make it into
    the playoffs, so the user wants watch the game in person. They want to
    look for hotels where the playoffs are occurring (GetNBANews +
    SearchHotels). After looking at the options, the user chooses to book a
    3-day stay at the cheapest 4-star option (BookHotel)."
    13

    This scenario exemplifies a scenario using 5 API calls. The scenario is
    complex, detailed, and concise as desired. The scenario also includes two
    APIs used in tandem, the required API, GetNBANews to search for the
    playoffs location and SearchHotels to find hotels based on the returned
    location. Usage of multiple APIs in tandem is highly desirable and will
    receive a higher score. Ideally each scenario should contain one or more
    instances of multiple APIs being used in tandem.

    Note that this scenario does not use all the APIs given and re-uses the "
    GetNBANews" API. Re-using APIs is allowed, but each scenario should
    involve as many different APIs as the user demands. Note that API usage is also included
    in the scenario, but exact parameters ar necessary. You must use a
    different combination of APIs for each scenario. All APIs must be used in
    at least one scenario. You can only use the APIs provided in the APIs
    section.
    
    Note that API calls are not explicitly mentioned and their uses are
    included in parentheses. This behaviour should be mimicked in your
    response.
    
    Output the tool usage in a strict json format with the function name and input to 
    the function. For example, Deliver your response in this format:
    
    ‘‘‘
    {tool_few_shot_examples}
    ‘‘‘
    # APIs
    ‘‘‘
    {tool_docs}
    ‘‘‘
    # Response
    ‘‘‘
    """
    return PROMPT
