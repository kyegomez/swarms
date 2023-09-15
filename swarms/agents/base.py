from abc import ABC, abstractmethod
from agent_protocol import Agent, Step, Task


class AbstractAgent:
    @staticmethod
    async def plan(step: Step) -> Step:
        task = await Agent.db.get_task(step.task_id)
        steps = generate_steps(task.input)

        last_step = steps[-1]
        for step in steps[:-1]:
            await Agent.db.create_step(
                task_id=task.task_id, 
                name=step, 
                pass
            )

        await Agent.db.create_step(
            task_id=task.task_id, 
            name=last_step, 
            is_last=True
        )
        step.output = steps
        return step

    @staticmethod
    async def execute(step: Step) -> Step:
        # Use tools, websearch, etc.
        ...

    @staticmethod
    async def task_handler(task: Task) -> None:
        await Agent.db.create_step(
            task_id=task.task_id, 
            name="plan", 
            pass
        )

    @staticmethod
    async def step_handler(step: Step) -> Step:
        if step.name == "plan":
            await AbstractAgent.plan(step)
        else:
            await AbstractAgent.execute(step)

        return step

    @staticmethod
    def start_agent():
        Agent.setup_agent(AbstractAgent.task_handler, AbstractAgent.step_handler).start()