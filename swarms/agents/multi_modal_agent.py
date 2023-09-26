from agent_protocol import Agent, Step, Task

from swarms.workers.multi_modal_workers.multi_modal_agent import MultiModalVisualAgent

class MultiModalVisualAgent:
    def __init__(
            self, 
            agent: MultiModalVisualAgent
        ):
        self.agent = agent
        self.plan = plan
    
    async def run(self, text: str) -> str:
        #run the multi-modal visual agent with the give task
        return self.agent.run_text(text)
    
    async def __call__(self, text: str) -> str:
        return self.agent.run(text)
    
    async def plan(self, step: Step) -> Step:
        task = Agent
        pass

    async def task_handler(self, task: Task):
        await self.agent.run()
    
    async def step_handler(self, step: Step):
        if step.name == "plan":
            await self.plan(step)
        else:
            await self.agent.run(step)

        return step
    
