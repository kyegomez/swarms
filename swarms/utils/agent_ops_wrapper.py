import logging
import agentops

class AgentOpsWrapper:
    """
    A wrapper for the AgentOps client that adds error handling and logging.
    """

    def __init__(self, api_key):
        """
        Initialize the AgentOps client with the given API key.
        """
        self.client = agentops.Client(api_key)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def record_action(self, action_name):
        """
        Record an action with the given name.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    self.client.record_action(action_name)
                    result = func(*args, **kwargs)
                    self.logger.info(f"Action {action_name} completed successfully.")
                    return result
                except Exception as e:
                    self.logger.error(f"Error while recording action {action_name}: {e}")
                    raise
            return wrapper
        return decorator

    def end_session(self, status):
        """
        End the session with the given status.
        """
        try:
            self.client.end_session(status)
            self.logger.info(f"Session ended with status {status}.")
        except Exception as e:
            self.logger.error(f"Error while ending session: {e}")
            raise
        
# agentops = AgentOpsWrapper(api_key)