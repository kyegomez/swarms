from swarms.agents.multi_modal_workers.multi_modal_agent import MultiModalVisualAgent
from swarms.agents.message import Message

class MultiModalAgent:
    """
    A user-friendly abstraction over the MultiModalVisualAgent that provides a simple interface 
    to process both text and images.
    
    Initializes the MultiModalAgent.

        Parameters:
            load_dict (dict, optional): Dictionary of class names and devices to load. Defaults to a basic configuration.
            temperature (float, optional): Temperature for the OpenAI model. Defaults to 0.
            default_language (str, optional): Default language for the agent. Defaults to "English".

    Usage
    --------------
    For chats:
    ------------
    agent = MultiModalAgent()
    agent.chat("Hello")

    -----------

    Or just with text
    ------------
    agent = MultiModalAgent()
    agent.run_text("Hello")

    
    """
    def __init__(
        self,
        load_dict,
        temperature,
        language: str = "english"
    ):
        self.load_dict = load_dict
        self.temperature = temperature
        self.langigage = language

        if load_dict is None:
            load_dict = {
                "ImageCaptioning": "default_device"
            }

        self.agent = MultiModalVisualAgent(
            load_dict,
            temperature
        )
        self.language = language
        self.history = []

    
    def run_text(
        self, 
        text: str = None, 
        language=None
    ):
        """Run text through the model"""

        if language is None:
            language = self.language

        try:
            self.agent.init_agent(language)
            return self.agent.run_text(text)
        except Exception as e:
            return f"Error processing text: {str(e)}"
    
    def run_img(
        self, 
        image_path: str, 
        language=None
    ):
        """If language is None"""
        if language is None:
            language = self.default_language
        
        try:
            return self.agent.run_image(
                image_path,
                language
            )
        except Exception as error:
            return f"Error processing image: {str(error)}"

    def chat(
        self,
        msg: str = None,
        language: str = None,
        streaming: bool = False
    ):
        """
        Run chat with the multi-modal agent
        
        Args:
            msg (str, optional): Message to send to the agent. Defaults to None.
            language (str, optional): Language to use. Defaults to None.
            streaming (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            str: Response from the agent
        
        Usage:
        --------------
        agent = MultiModalAgent()
        agent.chat("Hello")
        
        """
        if language is None:
            language = self.default_language

        #add users message to the history
        self.history.append(
            Message(
                "User",
                msg
            )
        )

        #process msg
        try:
            self.agent.init_agent(language)
            response = self.agent.run_text(msg)

            #add agent's response to the history
            self.history.append(
                Message(
                    "Agent",
                    response
                )
            )

            #if streaming is = True
            if streaming:
                return self._stream_response(response)
            else:
                response

        except Exception as error:
            error_message = f"Error processing message: {str(error)}"

            #add error to history
            self.history.append(
                Message(
                    "Agent",
                    error_message
                )
            )
            return error_message
    
    def _stream_response(
        self, 
        response: str = None
    ):
        """
        Yield the response token by token (word by word)
        
        Usage:
        --------------
        for token in _stream_response(response):
            print(token)
        
        """
        for token in response.split():
            yield token

    def clear(self):
        """Clear agent's memory"""
        try:
            self.agent.clear_memory()
        except Exception as e:
            return f"Error cleaning memory: {str(e)}"
    
