import uuid

class Task:
    def __init__(self, objective, priority=0, schedule=None, dependencies=None):
        self.id = uuid.uuid4()
        self.objective = objective
        self.priority = priority
        self.schedule = schedule
        self.dependencies = dependencies or []
        self.status = "pending"

