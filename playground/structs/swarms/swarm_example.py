from swarms import BaseSwarm, Agent, Anthropic


class MySwarm(BaseSwarm):
    def __init__(self, name="kyegomez/myswarm", *args, **kwargs):
        super(self, MySwarm).__init__(*args, **kwargs)
        self.name = name

        # Define and add your agents here
        self.agent1 = Agent(
            agent_name="Agent 1",
            system_prompt="A specialized agent for task 1.",
            llm=Anthropic(),
            max_loops=1,
            autosave=True,
            dashboard=False,
            streaming_on=True,
            verbose=True,
            stopping_token="<DONE>",
        )
        self.agent2 = Agent(
            agent_name="Agent 2",
            system_prompt="A specialized agent for task 2.",
            llm=Anthropic(),
            max_loops=1,
            autosave=True,
            dashboard=False,
            streaming_on=True,
            verbose=True,
            stopping_token="<DONE>",
        )
        self.agent3 = Agent(
            agent_name="Agent 3",
            system_prompt="A specialized agent for task 3.",
            llm=Anthropic(),
            max_loops=1,
            autosave=True,
            dashboard=False,
            streaming_on=True,
            verbose=True,
            stopping_token="<DONE>",
        )

    def run(self, task: str, *args, **kwargs):
        # Add your multi-agent logic here
        output1 = self.agent1.run(task, *args, **kwargs)
        output2 = self.agent2.run(task, output1, *args, **kwargs)
        output3 = self.agent3.run(task, output2, *args, **kwargs)
        return output3
