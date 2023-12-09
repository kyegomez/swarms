"""
Swarm Flow
Topic selection agent -> draft agent -> review agent -> distribution agent

Topic Selection Agent:
- Generate 10 topics on gaining mental clarity using Taosim and Christian meditation

Draft Agent:
- Write a 100% unique, creative and in human-like style article of a minimum of 5,000 words using headings and sub-headings.

Review Agent:
- Refine the article to meet PositiveMed’s stringent publication standards.

Distribution Agent:
- Social Media posts for the article.


# TODO
- Add shorter and better topic generator prompt
- Optimize writer prompt to create longer and more enjoyeable blogs
- Use Local Models like Storywriter


"""
from swarms.models import OpenAIChat
from termcolor import colored
