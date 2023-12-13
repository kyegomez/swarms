VISUAL_CHAIN_OF_THOUGHT = """
    
You, as the model, are presented with a visual problem. This could be an image containing various elements that you need to analyze, a graph that requires interpretation, or a visual puzzle. Your task is to examine the visual information carefully and describe your process of understanding and solving the problem.

Instructions:

Observation: Begin by describing what you see in the image. Break down the visual elements into understandable segments. For instance, if it's a picture of a street, identify the key components like cars, buildings, people, street signs, etc. If it's a graph, start by outlining its type, the axes, and the data it presents.

Initial Analysis: Based on your observation, start analyzing the image. If it's a scene, narrate the possible context or the story the image might be telling. If it's a graph or data, begin to interpret what the data might indicate. This step is about forming hypotheses or interpretations based on visual cues.

Detailed Reasoning: Delve deeper into your analysis. This is where the chain of thought becomes critical. If you're looking at a scene, consider the relationships between elements. Why might that person be running? What does the traffic signal indicate? For graphs or data-driven images, analyze trends, outliers, and correlations. Explain your thought process in a step-by-step manner.

Visual References: As you explain, make visual references. Draw arrows, circles, or use highlights in the image to pinpoint exactly what you're discussing. These annotations should accompany your verbal reasoning, adding clarity to your explanations.

Conclusion or Solution: Based on your detailed reasoning, draw a conclusion or propose a solution. If it's a visual puzzle or problem, present your answer clearly, backed by the reasoning you've just outlined. If it’s an open-ended image, summarize your understanding of the scene or the data.

Reflection: Finally, reflect on your thought process. Was there anything particularly challenging or ambiguous? How confident are you in your interpretation or solution, and why? This step is about self-assessment and providing insight into your reasoning confidence.

Example:

Let’s say the image is a complex graph showing climate change data over the last century.

Observation: "The graph is a line graph with time on the x-axis and average global temperature on the y-axis. There are peaks and troughs, but a general upward trend is visible."

Initial Analysis: "The immediate observation is that average temperatures have risen over the last century. There are fluctuations, but the overall direction is upward."

Detailed Reasoning: "Looking closer, the steepest increase appears post-1950. This aligns with industrial advancements globally, suggesting a link between human activity and rising temperatures. The short-term fluctuations could be due to natural climate cycles, but the long-term trend indicates a more worrying, human-induced climate change pattern."

Visual References: "Here [draws arrow], the graph shows a sharp rise. The annotations indicate major industrial events, aligning with these spikes."

Conclusion or Solution: "The data strongly suggests a correlation between industrialization and global warming. The upward trend, especially in recent decades, indicates accelerating temperature increases."

Reflection: "This analysis is fairly straightforward given the clear data trends. However, correlating it with specific events requires external knowledge about industrial history. I am confident about the general trend, but a more detailed analysis would require further data."    
    
    
"""
