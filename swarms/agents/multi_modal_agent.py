from swarms.agents.muti_modal_workers.multi_modal_agent import MultiModalVisualAgent

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
    
    def run_text(self, text, language=None):
        """Run text through the model"""

        if language is None:
            language = self.language

        try:
            self.agent.init_agent(language)
            return self.agent.run_text(text)
        except Exception as e:
            return f"Error processing text: {str(e)}"
    
    def run_img(self, image_path: str, language=None):
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
    
    def clear(self):
        try:
            self.agent.clear_memory()
        except Exception as e:
            return f"Error cleaning memory: {str(e)}"
    
