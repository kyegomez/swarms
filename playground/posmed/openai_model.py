from swarms.models.openai_models import OpenAIChat
import PosMedPrompts

openai = OpenAIChat(openai_api_key="sk-S4xHnFJu7juD33jxjJZfZU1cZYi")

draft = openai(PosMedPrompts.getDraftPrompt("AI in healthcare", "Pyschology"))


review = openai(PosMedPrompts.getReviewPrompt(draft))

print(review)
