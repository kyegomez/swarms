from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer
import pathlib

# Langchain
tool_name, tool_url = "gradio_tools", "http://127.0.0.1:8079/tools/gradio_tools/"

tool_name, tool_config = load_single_tools(tool_name, tool_url)

print(tool_name, tool_config)
stqa = STQuestionAnswerer()

agent = stqa.load_tools(tool_name, tool_config)

# test ToVideoTool
agent("Create a video for a panda eating bamboo on a rock.")

# test BarkTextToSpeechTool, StableDiffusionTool, StableDiffusionPromptGeneratorTool
# agent("Please create a jingle for a spanish company called 'Chipi Chups' that makes lollipops. "
#                           "The jingle should be catchy and playful and meant to appeal to all ages.")

# test DocQueryDocumentAnsweringTool,ImageCaptioningTool
IMG_PATH = pathlib.Path(__file__).parent / "florida-drivers-license.jpeg"
agent(
    f"Captioning the picture in {IMG_PATH}?The Action in langchain should follow the format:'Action:API's name'.For example,'Action:get_imagecaption.'"
)
# agent(f"Create music from picture in {IMG_PATH}?The Action in langchain should follow the format:'Action:API's name'.For example,'Action:get_imgtomsc.'")
# agent(f"What is the date of birth the driver in {IMG_PATH}?The Action in langchain should follow the format:'Action:API's name'.For example,'Action:get_qa.'")
