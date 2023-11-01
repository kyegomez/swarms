from swarms import OpenAIChat
from termcolor import colored

TOPIC_GENERATOR = f"""

First search for a list of topics on the web based their relevance to Positive Med's long term vision then rank than based on the goals this month, then output a single headline title for a blog for the next autonomous agent to write the blog, utilize the SOP below to help you strategically select topics. Output a single topic that will be the foundation for a blog.

VISION: Emphasis on exotic healthcare for improved health using Taoism, Ayurveda, and other ancient practices.

GOALS THIS MONTH: Clicks and engagement


Rank the topics and then return the most likely topic to satisfy the goals this month.


###########
Standard Operating Procedure for Topic Selection for PositiveMed.com

Objective: 
The goal of this SOP is to provide clear guidelines and best practices for selecting high-quality, engaging, and SEO-friendly topics to create content for PositiveMed.com. The content should align with PositiveMed's brand mission of providing valuable health, wellness, and medical information to readers. 

Overview:
Topic selection is a crucial first step in creating content for PositiveMed. Topics should inform, interest and engage readers, while also attracting search engine traffic through optimized keywords. This SOP covers core strategies and processes for researching, evaluating and selecting optimal topics.

Roles & Responsibilities:
The content team, consisting of writers, editors and content strategists, own the topic selection process.

The content team is responsible for:
- Monitoring health, medical, wellness trends and current events
- Conducting keyword research 
- Assessing site analytics and reader feedback
- Crowdsourcing topic ideas from internal team and external contributors
- Maintaining editorial calendar with upcoming topics
- Pitching and selecting topics for content approval

The editorial team is responsible for:
- Providing final approval on topics based on brand suitability, reader interest, and potential traffic/engagement 
- Ensuring selected topics are differentiated and not duplicative of existing content
- Reviewing and updating keyword opportunities tied to topics

Topic Sourcing
A strong content calendar begins with investing time into researching and generating promising topics. Here are key tactics and guidelines for sourcing topics:

Monitor Trends:
- Set Google Alerts for relevant keywords like "health news," "fitness trends," "nutrition research" etc. to receive daily updates.
- Subscribe to email newsletters, RSS feeds from authoritative sites like CDC, NIH, Mayo Clinic etc. 
- Follow social media accounts of health organizations and influencers to stay on top of latest discussions.
- Check online communities like Reddit, Quora, Facebook Groups for emerging topics.
- Look for real-world events, awareness months, holidays that tie into health observances.

Perform Keyword Research:  
- Use keyword research tools such as Google Keyword Planner, SEMrush, Moz Keyword Explorer etc.
- Target keywords with moderate-high search volume and low competition for the best opportunity.
- Look for conversational long-tail keywords that are more conversational and closely tied to topic themes. 
- Ensure keywords have not been over-optimized by competitors to avoid saturation.
- Aim for topics that offerClusters of interconnected keywords around related sub-topics. This allows targeting several keywords with one piece of content.

Analyze Site Analytics:
- Review Google Analytics data to identify:
- Most-read articles - Consider follow-up content or additional installments.
- Highest-traffic landing pages - Expand on topics driving site visitors.
- Top-performing categories - Prioritize related subjects that attract readers.
- Look for content gaps - Assess which categories have not been recently updated and need fresh content.

Crowdsource Topic Ideas:
- Ask readers to suggest topics through surveys, emails, social media, comments etc. 
- Review discussions in online communities to find topics readers are interested in.
- Collaborate with guest contributors who may pitch relevant ideas and angles. 
- Solicit insights from internal team members who interact closely with readers.

Map Editorial Calendar:
- Maintain a content calendar that maps topics over weeks and months. 
- Ensure a healthy mix of evergreen and trending topics across categories. 
- Balance informational articles with more entertaining listicles or quizzes.
- Schedule both individual articles and content series around specific themes.  
- Revisit calendar routinely to incorporate new topics as they emerge.

Evaluate Ideas
With a robust list of prospective topics, the next step is determining which ideas are worth pursuing. Use these criteria when assessing the merit of topics:

Reader Interest:
- Would the topic pique the curiosity of PositiveMed's target audience?
- Does it address questions readers may be asking about health, medicine, nutrition?
- Will it appeal to readers' needs for wellness tips, self-improvement advice?
- Does it present an interesting angle on a known subject versus just reporting basic facts?

Differentiation:
- Has this specific topic been recently covered on PositiveMed or similar sites? 
- If covered before, does the pitch offer a novel spin - new research, fresh data, contrarian view?
- Will the content provide value-add beyond what readers can easily find through a Google search?

Brand Suitability:  
- Does the topic match the tone and mission of the PositiveMed brand?
- Will the content uphold PositiveMed's standards for accuracy, credibility and ethics?
- Could the topic be construed as promoting unproven advice or "pseudoscience"?

Positioning:
- What unique perspective can PositiveMed bring that differs from mainstream health sites?
- Does the topic lend itself to an uplifting, empowering message aligned with the brand?
- Can the material be framed in a way that resonates with PositiveMed's niche audience? 

Actionability: 
- Will readers come away with new knowledge they can apply in their daily lives?
- Can the content offer clear steps, takeaways for improving health and wellbeing?
- Does the topic present opportunities to include tips, product recommendations etc.?

Timeliness:
- Is this tied to a recent news event or emerging trend that warrants timely coverage?
- For evergreen topics, are there new studies, pop culture references etc. that can make it timely?
- Does the angle offer a way to make an old topic feel fresh and relevant?

Competition:
- How saturated is the topic market? Who has top-ranking content on this topic?
- Does PositiveMed have a strong opportunity to own the conversation with a unique take?
- What value can be added versus competitor content on this subject?

Commercial Viability: 
- Does the topic allow integrating affiliate links, product recommendations, lead generation offers etc.?
- Can it support the development of related products or paid offerings in the future?
- Will it attract engagement and social shares to increase traffic?

Keyword Integration 

With promising topics identified, the next step is integrating keywords into content plans and outlines. 

Conduct Keyword Research:
- Identify primary target keyword for topic that has:
- Moderate-to-high search volume 
- Low-to-medium competition
- Relevance to topic and PositiveMed's niche

Find Supporting Keywords:  
- Build a cluster of 3-5 secondary keywords around topic including:
- Related searches and questions
- Semantically connected words/phrases  
- Keyword variations (long tail, alternate wording etc.)
- Stay within minimum monthly search volumes

Map Out Keywords:
- Determine optimal keyword placement for outlined sections e.g.:
- Primary KW in title, H1, intro, conclusion
- Supporting KWs in H2s, first sentence of paras etc.
- Include keywords naturally - no over-optimization

Check Cannibalization:  
- Compare suggested keywords against existing content to avoid targeting same terms.
- Modify keywords if needed to differentiate and drive incremental traffic.

Review Opportunities:
- Cross-check keywords in planning tools to confirm search volume and competition.
- Align keywords with buyer intent and top of funnel to mid funnel searches.
- Ensure keywords are entered into analytics to track conversions.

Style and Tone Guidelines

In line with PositiveMed's brand voice, content should adopt an:

Educational yet conversational tone:
- Explain health topics, science and research simply without over-simplifying complex issues. 
- Present insightful information in a way that is accessible and engaging for a layperson audience.

Empowering and motivational style:
- Frame content with an uplifting, inspirational tone versus fear-mongering or alarming portrayal of health risks.
- Provide encouraging advice to inspire readers to take charge of their wellbeing.

Trustworthy and ethical approach:
- Uphold highest standards of accuracy, credibility and reliability.
- Cite legitimate sources. Avoid promoting unverified claims or exaggerated benefits.
- Disclose risks, drawbacks and limitations of health approaches covered.

Inclusive and compassionate voice:  
- Reflect diversity and sensitivity towards people of different backgrounds, conditions and needs.  
- Consider circumstances like financial constraints, disabilities, cultural values etc. that impact health choices.

Hopeful outlook grounded in facts:
- Focus on solutions and a positive outlook while still being realistic.
- Counter misinformation; clarify myths vs facts.
"""

DRAFT_PROMPT = """# MISSION
Write a 100% unique, creative and in human-like style article of a minimum of 5,000 words using headings and sub-headings.
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
You should write a conclusion end with either a call to action or a clear and concise point. What is the key takeaway?
"""


REVIEW_PROMPT = """# MISSION
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


SOCIAL_MEDIA_SYSTEM_PROMPT_AGENT = """
You're the Social Media System Agent. Your job is to create a social media post for the article below.

Your responsibilities are:
Publishing and Distribution:
    •    Publishing AI Agent:
    •    Automated publishing to designated platforms.
    •    Formatting checks for platform compatibility.
    •    Distribution:
    •    Automated sharing to social media channels.
    •    Email distribution to subscriber list.


Create 3 high converting posts for instagram, facebook, twitter, linkedin, and pinterest optimizing for {{GOAL}}

# ARTICLE:
{{ARTICLE}}


"""

llm = OpenAIChat(openai_api_key="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")


def get_draft_prompt(topic, theme):
    prompt = DRAFT_PROMPT.replace("{{TOPIC}}", topic).replace("{{THEME}}", theme)
    return prompt


def get_review_prompt(article):
    prompt = REVIEW_PROMPT.replace("{{ARTICLE}}", article)
    return prompt


def social_media_prompt(article: str, goal: str = "Clicks and engagement"):
    prompt = SOCIAL_MEDIA_SYSTEM_PROMPT_AGENT.replace("{{ARTICLE}}", article).replace(
        "{{GOAL}}", goal
    )
    return prompt

# Agents
topic_selection_task = (
    "Generate 10 topics on gaining mental clarity using Taosim and Christian meditation"
)
topic_selection_agent = llm(
    f"Your System Instructions: {TOPIC_GENERATOR}, Your current task: {topic_selection_task}"
)
dashboard = print(
    colored(
        f"""
    Topic Selection Agent
    -----------------------------

    Topics:
    ------------------------
    {topic_selection_agent}
    
    """,
        "blue",
    )
)


draft_agent = llm(get_draft_prompt(topic_selection_agent, topic_selection_agent))
draft_out = print(
    colored(
        f"""
    Drafter Writer Agent
    -----------------------------

    Draft:
    ------------------------
    {topic_selection_agent}
    
    """,
        "Green",
    )
)
print(draft_out)


# Agent that reviews the draft
review_agent = llm(get_review_prompt(draft_agent))
print(review_agent)


# Agent that publishes on social media
distribution_agent = llm(
    social_media_prompt(draft_agent, goal="Clicks and engagement")
)
distribution_agent_out = print(
    colored(
        f"""
        Distribution Agent
        -------------------

        Social Media Posts
        -------------------
        {distribution_agent}

        """
    )
)
print(distribution_agent_out)