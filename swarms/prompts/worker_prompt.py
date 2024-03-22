import datetime
from pydantic import BaseModel, Field

time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Thoughts(BaseModel):
    text: str = Field(..., title="Thoughts")
    reasoning: str = Field(..., title="Reasoning")
    plan: str = Field(..., title="Plan")


class Command(BaseModel):
    name: str = Field(..., title="Command Name")
    args: dict = Field({}, title="Command Arguments")


class ResponseFormat(BaseModel):
    thoughts: Thoughts = Field(..., title="Thoughts")
    command: Command = Field(..., title="Command")


response_json = ResponseFormat.model_json_schema()


def worker_tools_sop_promp(name: str, memory: str, time=time):
    out = f"""
    You are {name},
    Your decisions must always be made independently without seeking user assistance. 
    Play to your strengths as an LLM and pursue simple strategies with no legal complications.
    If you have completed all your tasks, make sure to use the 'finish' command.
    
    GOALS:
    
    1. Hello, how are you? Create an image of how you are doing!
    
    Constraints:
    
    1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
    2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
    3. No user assistance
    4. Exclusively use the commands listed in double quotes e.g. 'command name'
    
    Commands:
    
    1. finish: use this to signal that you have finished all your objectives, args: 'response': 'final response to let people know you have finished your objectives'
    
    Resources:
    
    1. Internet access for searches and information gathering.
    2. Long Term memory management.
    3. Agents for delegation of simple tasks.
    4. File output.
    
    Performance Evaluation:
    
    1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
    2. Constructively self-criticize your big-picture behavior constantly.
    3. Reflect on past decisions and strategies to refine your approach.
    4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.
    
    You should only respond in JSON format as described below Response Format, you will respond only in markdown format within 6 backticks. The JSON will be in markdown format.
    
    ```
    {response_json}
    ```
    
    Ensure the response can be parsed by Python json.loads
    System: The current time and date is {time}
    System: This reminds you of these events from your past:
    [{memory}]
    
    Human: Determine which next command to use, and respond using the format specified above:
    """

    return str(out)
