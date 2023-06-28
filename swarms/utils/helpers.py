class WorkerNode:
    def __init__(self, llm: BaseLLMAgent):
        self.llm = llm
        self.task_queue = deque()
        self.completed_task = deque()

    def receieve_task(self, task):
        self.task_queue.append(task)

    def complete_task(self):
        task = self.task_queue.popleft()
        result = self.llm.execute(task)
        self.completed_task.append(result)
        return result
    
    def communicates(self):
        task = self.task_queue.popleft()
        result = self.llm.execute(task)
        self.completed_tasks.append(result)
        return result
    
    def communicate(self, other_node):
        #palceholder for communication method which is utilizing an omni -present Ocean instance
        pass


class Swarms:
    def __init__(self, num_nodes: int, llm: BaseLLM, self_scaling: bool): 
        self.nodes = [WorkerNode(llm) for _ in range(num_nodes)]
        self.self_scaling = self_scaling
    
    def add_worker(self, llm: BaseLLM):
        self.nodes.append(WorkerNode(llm))

    def remove_workers(self, index: int):
        self.nodes.pop(index)

    def execute(self, task):
        #placeholer for main execution logic
        pass

    def scale(self):
        #placeholder for self scaling logic
