from simpleaichat import AIChat

class OpenAI:
    def __init__(self,
                 api_key=None,
                 system=None,
                 console=True,
                 model=None,
                 params=None,
                 save_messages=True):
        self.api_key = api_key or self._fetch_api_key()
        self.system = system or "You are a helpful assistant"
        self.ai = AIChat(api_key=self.api_key, 
            system=self.system, 
            console=console, 
            model=model, 
            params=params, 
            save_messages=save_messages)
    
    def generate(self, message, **kwargs):
        try:
            return self.ai(message, **kwargs)
        except Exception as error:
            print(f"Error in OpenAI, {error}")
    
    def fetch_api_key(self):
        pass


#usage
#from swarms import OpenAI()
#chat = OpenAI()
#response = chat("Hello World")
#print(response)

