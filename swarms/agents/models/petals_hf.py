
class PetalsHFLLM:
    def __init__(self, model_name: str = None, prompt: str = None, device: str = None, use_fast = False, add_bos_token: str = None, cuda=False):
        self.model_name = model_name
        self.prompt = prompt 
        self.device = device
        self.use_fast = use_fast
        self.add_bos_token = add_bos_token
        self.cuda = cuda