from swarms.worker.worker import Worker

class SimpleSwarm:
    def __init__(
       self,
       num_workers,
        openai_api_key,
        ai_name
    ):
        """
        
        # Usage
        swarm = Swarm(num_workers=5, openai_api_key="", ai_name="Optimus Prime")
        task = "What were the winning Boston Marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
        responses = swarm.distribute_task(task)

        for response in responses:
            print(response)
        
        """
        self.workers = [
            Worker(openai_api_key, ai_name) for _ in range(num_workers)
        ]

    def run(self, task):
        responses = []
        for worker in self.workers:
            response = worker.run(task)
            responses.append(response)
        return responses
    
    def __call__(self, task):
        return self.run(task) 
    