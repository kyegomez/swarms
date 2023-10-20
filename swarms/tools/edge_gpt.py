from swarms.tools.tool import BaseTool


class EdgeGPTTool(BaseTool):
    def __init__(
        self,
        model,
        name="EdgeGPTTool",
        description="Tool that uses EdgeGPTModel to generate responses",
    ):
        super().__init__(name=name, description=description)
        self.model = model

    def _run(self, prompt):
        return self.model.__call__(prompt)
