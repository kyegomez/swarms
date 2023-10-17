

#base App class
class App:
    """
    This is a base app class for examples

    Args:
    worker: Worker Agent

    Usage

    app = App(Worker)
    app.run()

    """
    def __init__(
        self,
        worker, 
    ):
        self.worker = worker
        self.worker.app = self
    
    def run(self, task):
        """Run the app"""
        pass
