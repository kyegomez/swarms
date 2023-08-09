#kye 
#aug 8, 11:51

from simpleaichat import AIChat, AsyncAIChat
import asyncio


class OpenAI:
    def __init__(self,
                 api_key=None,
                 system=None,
                 console=True,
                 model=None,
                 params=None,
                 save_messages=True):
        self.api_key = api_key or self.fetch_api_key()
        self.system = system or "You are a helpful assistant"

        try:
                
            self.ai = AIChat(api_key=self.api_key, 
                system=self.system, 
                console=self.console, 
                model=self.model, 
                params=self.params, 
                save_messages=self.save_messages)
            
            self.async_ai = AsyncAIChat(
                api_key=self.api_key,
                system=self.system,
                console=self.console,
                model=self.model,
                params=self.params,
                save_messages=self.save_messages
            )
        
        except Exception as error:
            raise ValueError(f"Failed to initialize the chat with error: {error}, check inputs and input types")
    
    def __call__(self, message, **kwargs):
        try:
            return self.ai(message, **kwargs)
        except Exception as error:
            print(f"Error in OpenAI, {error}")
    
    def generate(self, message, **kwargs):
        try:
            return self.ai(message, **kwargs)
        except Exception as error:
            print(f"Error in OpenAI, {error}")
        
    async def generate_async(self, message, **kwargs):
        try:
            return await self.async_ai(message, **kwargs)
        except Exception as error:
            raise Exception(f"Error in asynchronous OpenAI Call, {error}")
    
    def initialize_chat(self, ids):
        for id in ids:
            try:
                self.async_ai.new_session(api_key=self.api_key, id=id)
            except Exception as error:
                raise ValueError(f"Failed to initialize session for ID {id} with error: {error}")
    
    async def ask_multiple(self, ids, question_template):
        try:
            self.initialize_chat(ids)
            tasks = [self.async_ai(question_template.format(id=id), id=id) for id in ids]
            return await asyncio.gather(*tasks)
        except Exception as error:
            raise Exception(f"Error in ask_multiple: method: {error}")
    
    async def stream_multiple(self, ids, question_template):
        try:
            self.initialize_chat(ids)

            async def stream_id(id):
                async for chunk in await self.async_ai.stream(question_template.format(id=id), id=id):
                    response = chunk["response"]
                return response
        
            tasks = [stream_id(id) for id in ids]
            return await asyncio.gather(*tasks)
        except Exception as error:
            raise Exception(f"Error in stream_multiple method: {error}")
    
    def fetch_api_key(self):
        pass


#usage
#from swarms import OpenAI()
#chat = OpenAI()
#response = chat.generate("Hello World")
#print(response)

#async
# async_responses = asyncio.run(chat.ask_multiple(['id1', 'id2'], "How is {id}"))
# print(async_responses)
