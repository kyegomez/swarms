import asyncio
from typing import Optional, Callable, Any

class Agent:
    def __init__(self, 
                 name: str, 
                 max_loops: int = 10, 
                 streaming_on: bool = False,
                 on_token: Optional[Callable[[str], None]] = None):
        self.name = name
        self.max_loops = max_loops
        self.streaming_on = streaming_on
        self.on_token = on_token
        self.loop_count = 0
        self.summary = ""
        
    async def plan(self) -> str:
        """Generate a plan for the agent"""
        plan = f"Plan for {self.name}"
        if self.streaming_on and self.on_token:
            for char in plan:
                self.on_token(char)
                await asyncio.sleep(0.01)
        return plan
    
    async def execute(self, plan: str) -> str:
        """Execute the plan"""
        execution = f"Executing: {plan}"
        if self.streaming_on and self.on_token:
            for char in execution:
                self.on_token(char)
                await asyncio.sleep(0.01)
        return execution
    
    async def generate_summary(self, plan: str, execution: str) -> str:
        """Generate a summary of the plan and execution"""
        summary = f"Summary: {plan} - {execution}"
        if self.streaming_on and self.on_token:
            for char in summary:
                self.on_token(char)
                await asyncio.sleep(0.01)
        return summary
    
    async def autonomous_loop(self):
        """Main autonomous loop for the agent"""
        while self.loop_count < self.max_loops:
            plan = await self.plan()
            execution = await self.execute(plan)
            self.loop_count += 1
            
            if self.loop_count >= self.max_loops:
                self.summary = await self.generate_summary(plan, execution)
                break
    
    def get_summary(self) -> str:
        """Get the final summary"""
        return self.summary
```

```python