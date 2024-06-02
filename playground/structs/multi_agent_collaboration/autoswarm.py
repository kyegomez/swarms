from swarms import BaseSwarm, AutoSwarmRouter


class FinancialReportSummarization(BaseSwarm):
    def __init__(self, name: str = None, *args, **kwargs):
        super().__init__()

    def run(self, task, *args, **kwargs):
        return task


# Add swarm to router
router = AutoSwarmRouter(swarms=[FinancialReportSummarization])

# Create AutoSwarm Instance
autoswarm = AutoSwarmRouter(
    name="kyegomez/FinancialReportSummarization",
    description="A swarm for financial document summarizing and generation",
    verbose=True,
    router=router,
)

# Run the AutoSwarm
autoswarm.run("Analyze these documents and give me a summary:")
