from swarms import AutoSwarm, AutoSwarmRouter, BaseSwarm


# Build your own Swarm
class MySwarm(BaseSwarm):
    def __init__(self, name="kyegomez/myswarm", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def run(self, task: str, *args, **kwargs):
        # Add your multi-agent logic here
        # agent 1
        # agent 2
        # agent 3
        return "output of the swarm"


# Add your custom swarm to the AutoSwarmRouter
router = AutoSwarmRouter(
    swarms=[MySwarm]
)


# Create an AutoSwarm instance
autoswarm = AutoSwarm(
    name="kyegomez/myswarm",
    description="A simple API to build and run swarms",
    verbose=True,
    router=router,
)


# Run the AutoSwarm
autoswarm.run("Analyze these financial data and give me a summary")
