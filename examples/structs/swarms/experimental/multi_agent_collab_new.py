from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.agent import Agent
from typing import Optional, Sequence, List, Dict, Any, Coroutine
from swarms_memory import BaseVectorDatabase


class FederatedSwarm(BaseSwarm):
    def __init__(
        self,
        name: Optional[str] = "FederatedSwarm",
        description: Optional[str] = "A swarm of swarms",
        swarms: Optional[Sequence[BaseSwarm]] = None,
        memory_system: Optional[BaseVectorDatabase] = None,
        max_loops: Optional[int] = 4,
        *args,
        **kwargs,
    ):
        super().__init__(
            name=name, description=description, *args, **kwargs
        )
        self.name = name
        self.description = description
        self.swarms: List[BaseSwarm] = swarms if swarms is not None else []
        self.memory_system = memory_system
        self.max_loops = max_loops
        self.shared_memory: Dict[str, Any] = {} # For inter-swarm communication

    def add_swarm(self, swarm: BaseSwarm):
        self.swarms.append(swarm)

    def remove_swarm(self, swarm: BaseSwarm):
        if swarm in self.swarms:
            self.swarms.remove(swarm)

    def get_swarm(self, name: str) -> Optional[BaseSwarm]:
        for swarm in self.swarms:
            if swarm.name == name:
                return swarm
        return None

    def get_swarm_agents(self) -> List[Agent]:
        agents = []
        for swarm in self.swarms:
            agents.extend(swarm.agents)
        return agents

    def get_swarm_agent(self, name: str) -> Optional[Agent]:
        for agent in self.get_swarm_agents():
            if agent.agent_name == name:
                return agent
        return None


    async def run_single_swarm(
        self, swarm: BaseSwarm, task: str, **kwargs
    ) -> Any:
        return await swarm.run(task, **kwargs)

    async def run_multiple_swarms_parallel(self, task: str, **kwargs) -> List[Any]:
        async def run_swarm_with_shared_memory(swarm):
            # Inject shared memory into the swarm's kwargs if the swarm needs it
            swarm_kwargs = kwargs.copy()
            if hasattr(swarm, "shared_memory"):
                 swarm.shared_memory = self.shared_memory
            return await self.run_single_swarm(swarm, task, **swarm_kwargs)

        tasks = [run_swarm_with_shared_memory(swarm) for swarm in self.swarms]
        results = await asyncio.gather(*tasks)
        return results

    async def run_multiple_swarms_sequential(self, task:str, **kwargs) -> List[Any]:
        results = []
        for swarm in self.swarms:
            swarm_kwargs = kwargs.copy()
            if hasattr(swarm, "shared_memory"):
                 swarm.shared_memory = self.shared_memory
            result = await self.run_single_swarm(swarm, task, **swarm_kwargs)
            results.append(result)
            # Optionally update the task based on previous swarm's output
            # For sequential execution, you can chain outputs:
            task = result # Or some function of the result

        return results


    async def run(self, task: str, parallel: bool = True, **kwargs):
        if parallel:
            results = await self.run_multiple_swarms_parallel(task, **kwargs)
        else:
            results = await self.run_multiple_swarms_sequential(task, **kwargs)

        return results
