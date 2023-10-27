draftPrompt = """# MISSION
Write a 100% unique, creative and in human-like style article of a minimum of 2000 words using headings and sub-headings.
Ensure your tone is Professional and casual while focusing on presenting information and analysis without excessive embellishment.

The topic is: {{TOPIC}}

# GENERAL GUIDELINE
There should be minimum 15 headings and 10 sub-headings in the content. 
Try to write at least 300-400 words of content for each heading or sub-heading. bold all the headings and sub-headings using Markdown formatting. 
Try to use contractions, idioms, transitional phrases, interjections, dangling modifiers, and colloquialisms, and avoid repetitive phrases and unnatural sentence structures. 
When you write, you will correctly format the blog post according to proper SEO standards, with as much rich and detailed HTML as possible, for example, lists, bold, italics, quotes from the internet, tables, and external links to high-quality websites such as Wikipedia.
Try to ask questions and then immediately give a good and concise answer, to try to achieve the featured snippet on Google.
The article should include SEO meta-description (must include the [PROMPT] in the description), an Introduction, and a click-worthy short title. Also, Use the seed keyword as the first H2.
Always use a combination of paragraphs, lists, and tables for a better reader experience. 
Write at least one paragraph with the heading {{TOPIC}}. 
Try not to change the original {{TOPIC}} while writing the Title.
Try to mention "{{TOPIC}}" 2-3 times in the article.
Write content that can easily pass the AI detection tools test.
You should write a conclusion end with either a call to action or a clear and concise point. What is the key takeaway?"""


reviewPrompt = """# MISSION
You are responsible for refining an article to meet PositiveMed’s stringent publication standards. 
Your role involves content analysis, editorial precision, expert validation, legal verification, and overall quality assurance. 

# ContentReview:
- Provide constructive feedback on outline and drafts content 
- Collect input on strengths to leverage and areas needing improvement.

# Editor Review:  
- Evaluate initial drafts for errors, gaps that require additional research.
- Provide guidance on better organizing structure and flow.
- Assess tone, voice and brand alignment.

# Expert Review:
- Ask medical experts related to article topic to validate accuracy of information.
- Verify advice follows ethical guidelines accepted by the medical community.   
- Request quotes that lend credibility and reinforce key points.

# Legal Review:  
- Confirm content meets regulatory standards for health claims and liability risks.
- Address any recommended edits to mitigate brand reputation risk.

# Quality Checklist:  Scrutinize final draft against PositiveMed's standards:
- Medical accuracy - error-free facts/statistics, supported claims 
- Logical flow - smooth transitions, complementary sections  
- Reader value - insightful analysis beyond fluffy content
- Brand alignment - uplifting tone, inclusive messaging
- Strong conclusion - memorable takeaways, relevant next steps/resources for readers

# ARTICLE TO REVIEW:
{{ARTICLE}}

# OUTPUT:
Re-Write the article, taking into account all review instructions and standards
"""

def getDraftPrompt(topic, theme):
    prompt = draftPrompt.replace("{{TOPIC}}", topic).replace("{{THEME}}", theme)
    return prompt

def getReviewPrompt(article):
    prompt = reviewPrompt.replace("{{ARTICLE}}", article)
    return prompt
