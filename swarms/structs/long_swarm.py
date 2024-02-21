from typing import List

from swarms.structs.agent import Agent
from swarms.utils.parse_code import extract_code_from_markdown


class LongContextSwarmLeader:
    """
    Represents a leader in a long context swarm.

    Args:
        - llm (str): The language model to use for the agent.
        - agents (List[Agent]): The agents in the swarm.
        - prompt_template_json (str): The SOP template in JSON format.
        - return_parsed (bool): Whether to return the parsed output.
    
    """

    def __init__(
        self,
        llm,
        agents: List[Agent] = None,
        prompt_template_json: str = None,
        return_parsed: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.agents = agents
        self.prompt_template_json = prompt_template_json
        self.return_parsed = return_parsed
        
        # Create an instance of the Agent class
        self.agent = Agent(
            llm=llm,
            system_prompt=None,
            sop=self.prompt_template_json, 
            *args, 
            **kwargs
        )

        
    def prep_schema(self, task: str, *args, **kwargs):
        """
        Returns a formatted string containing the metadata of all agents in the swarm.

        Parameters:
        - task (str): The description of the task.

        Returns:
        - prompt (str): The formatted string containing the agent metadata.
        """
        prompt = f"""
        
        You need to recruit a team of members to solve a
        task. Select the appropriate member based on the
        task description:
        
        # Task Description
        {task}
        
        # Members
        
        Your output must follow this JSON schema below in markdown format:
            {{
                "agent_id": "string",
                "agent_name": "string",
                "agent_description": "string"
            }}
        
        """
        for agent in self.agents:
            prompt += f"Member Name: {agent.ai_name}\nMember ID: {agent.id}\nMember Description: {agent.description}\n\n"
        
        return prompt
    
    
    def prep_schema_second(
        self,
        task_description: str,
        task: str
    ):
        prompt = f"""
        You are the leader of a team of {len(self.agents)}
        members. Your team will need to collaborate to
        solve a task. The rule is:
        
        1. Only you know the task description and task
        objective; the other members do not.
        2. But they will receive different documents that
        may contain answers, and you need to send them
        an instruction to query their document.
        3. Your instruction need to include your
        understanding of the task and what you need them
        to focus on. If necessary, your instructions can
        explicitly include the task objective.
        4. Finally, you need to complete the task based on
        the query results they return.
        
        # Task Description:
        {task_description}
        
        # Task Objective:
        {task}
        
        # Generate Instruction for Members:
        Now, you need to generate an instruction for all
        team members. You can ask them to answer a
        certain question, or to extract information related
        to the task, based on their respective documents.
        Your output must following the JSON
        format: {{"type": "instruction", "content":
        "your_instruction_content"}}
        
        """
        return prompt
    

    def run(self, task: str, *args, **kwargs):
        """
        Executes the specified task using the agent's run method.

        Args:
            task: The task to be executed.
            *args: Additional positional arguments for the task.
            **kwargs: Additional keyword arguments for the task.

        Returns:
            The result of the task execution.
        """
        task = self.prep_schema(task)
        out = self.agent.run(task, *args, **kwargs)
        
        if self.return_parsed:
            out = extract_code_from_markdown(out)
            
        return out

# class LongContextSwarm(BaseSwarm):
#     def __init__(
#         self,
#         agents: List[Agent],
#         Leader: Agent,
#         team_loops: int,
#         *args,
#         **kwargs,
#     ):
#         super().__init__()
#         self.agents = agents
#         self.leader = Leader
#         self.team_loops = team_loops
#         self.chunks = len(agents)
