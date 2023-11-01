"""
Flow

topic selection agent -> draft agent -> review agent -> distribution agent


Topic Selection Agent:
- Generate 10 topics on gaining mental clarity using Taosim and Christian meditation

Draft Agent:
- Write a 100% unique, creative and in human-like style article of a minimum of 5,000 words using headings and sub-headings.

Review Agent:
- Refine the article to meet PositiveMed’s stringent publication standards.



"""

from swarms import OpenAIChat
from termcolor import colored

TOPIC_GENERATOR = f"""

First search for a list of topics on the web based their relevance to Positive Med's long term vision then rank than based on the goals this month, then output a single headline title for a blog for the next autonomous agent to write the blog, utilize the SOP below to help you strategically select topics. Output a single topic that will be the foundation for a blog.

VISION: Emphasis on exotic healthcare for improved health using Taoism, Ayurveda, and other ancient practices.

GOALS THIS MONTH: Clicks and engagement


Rank the topics on a scale from 0.0 to 1.0 on how likely it is to achieve the goal and then return the single most likely topic to satisfy the goals this month.



########### Standard Operating Procedure for Topic Selection for PositiveMed.com  ######################

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






REVIEW_PROMPT = """

################ Your MISSION ##############################
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
You're the Social Media System Agent. Your job is to create social media posts for the article below.

Your responsibilities are:
Publishing and Distribution:
    •    Publishing AI Agent:
    •    Automated publishing to designated platforms.
    •    Formatting checks for platform compatibility.
    •    Distribution:
    •    Automated sharing to social media channels.
    •    Email distribution to subscriber list.


Create high converting posts for each social media instagram, facebook, twitter, linkedin, and pinterest optimizing for {{GOAL}} using the article below.

Denote the social media's by using the social media name in HTML like tags

<FACEBOOK> POST </FACEBOOK>
<TWITTER> POST </TWITTER>
<INSTAGRAM> POST </INSTAGRAM>

######### ARTICLE #######
{{ARTICLE}}
"""

llm = OpenAIChat(openai_api_key="")

def get_review_prompt(article):
    prompt = REVIEW_PROMPT.replace("{{ARTICLE}}", article)
    return prompt


def social_media_prompt(article: str, goal: str = "Clicks and engagement"):
    prompt = SOCIAL_MEDIA_SYSTEM_PROMPT_AGENT.replace("{{ARTICLE}}", article).replace(
        "{{GOAL}}", goal
    )
    return prompt

# Agent that generates topics
topic_selection_task = (
    "Generate 10 topics on gaining mental clarity using ancient Taosim practices"
)
topics = llm(
    f"Your System Instructions: {TOPIC_GENERATOR}, Your current task: {topic_selection_task}"
)

dashboard = print(
    colored(
        f"""
    Topic Selection Agent
    -----------------------------

    Topics:
    ------------------------
    {topics}
    
    """,
        "blue",
    )
)
# print(dashboard)

# Agent that generates blogs

DRAFT_AGENT_SYSTEM_PROMPT = f"""
Write a 5,000 word + narrative essay on a 100% unique, creative and in human-like style article of a minimum of 5,000 words using headings and sub-headings.
Ensure your tone is Professional and casual while focusing on presenting information and analysis without excessive embellishment.

Here is a list of topics, write the narrative on the first one: {topics}

############ Here is a 5,000+ word SOP on how to write high-quality articles for PositiveMed.com: #######################

Standard Operating Procedure for Article Writing for PositiveMed.com

Objective:
This SOP provides clear guidelines and best practices for crafting informative, engaging, SEO-optimized articles for PositiveMed.com. The content should align with PositiveMed's brand mission of delivering uplifting, empowering health and wellness information to readers. 

Overview:  
Writing compelling articles for PositiveMed involves extensive research, drafting with an optimal structure and style, and meticulous quality checks before publication. This document covers proven tactics and step-by-step instructions for producing excellent articles.

Roles & Responsibilities:

The content team owns the article writing process including:

- Conducting research 
- Developing outlines
- Writing original drafts
- Incorporating editor/peer feedback
- Fact-checking and proofreading  
- Ensuring brand consistency

The editorial team is responsible for:

- Providing input on outlines and drafts
- Reviewing for quality standards 
- Verifying tone aligns with brand voice
- Approving final drafts for publication
- Tracking performance analytics
        
Research Process

Thorough research is crucial for creating authoritative, evidence-based articles. Here are key research steps:

Set Parameters:
- Clarify objective, scope, length and deadline for article. This guides how extensive research should be.
- Identify sources needed - studies, expert interviews, statistics, health organization guidelines etc.

Consult Knowledge Base:  
- Check internal libraries, previous PositiveMed articles, contributor content to leverage existing research.
- Assess what background information can be summarized vs. new research required.

Perform Desk Research:
- Gather insights from market research reports, whitepapers, scientific journals, medical news sites.
- Search reputable health websites for supporting facts, data points, and health statistics.

Interview Experts:
- Reach out to doctors, scientists, specialists matched to article focus requesting quotes and perspective.
- Ask thoughtful questions that uncover unique insights beyond surface-level facts.

Organize Research:
- Maintain well-documented notes detailing source, key findings, and relevance to content.  
- Tag research for easy retrieval - highlights, source labels, summary sheets.
- Cite sources to track reference and substantiate claims.  

Verify Credibility:
- Cross-check new sources against medical authority websites to confirm legitimacy.
- Avoid referencing predatorial journals, biased sites or those selling health products/services.  
- Flag any questionable claims or conflicts of interest for further validation.

Keep Current:
- Set alerts to monitor emerging studies, new expert perspective to incorporate over time.  
- Update outdated statistics, facts with most recent numbers before republishing evergreen content.

Drafting Process 

With research completed, the next phase is developing an insightful article draft. Key steps include:

Outline Content:
- Organize research into logical sections and subsections for smooth flow.  
- Ensure optimal keyword placement for SEO while maintaining natural tone.
- Structure content to focus on most valuable information upfront.

Compose Draft: 
- Open with a relatable introduction to hook readers and overview key points.
- Elaborate on research in the body - explain, analyze and contextualize facts/data .
- Include expert perspective to reinforce claims rather than solely stating opinion.
- Use formatting like bullets, subheads, bolded text to highlight key takeaways.

Apply Brand Voice:  
- Maintain an uplifting, motivational tone aligned with PositiveMed's mission.  
- Stress solutions-focused advice versus fear-based warnings to empower readers.
- Use inclusive language and culturally sensitive medical references.

Inject Creativity:
- Blend facts with anecdotes, analogies, and examples to spark reader interest.
- Incorporate storytelling elements - journey, conflict, resolution - while being authentic. 
- Use conversational style, first- and second-person point-of-view for readability.

Check Accuracy: 
- Verify all medical statements against legitimate sources like CDC, Mayo Clinic, NIH.
- Scrutinize cited data for relevance and statistical significance.  
- Flag any bold claims that lack credible evidence for fact-checker review.   

Review Process

Before publication, all PositiveMed articles should go through multiple review rounds.

Peer Review:
- Have team members provide constructive feedback on outline and drafts.  
- Collect input on strengths to leverage and areas needing improvement.

Editor Review:  
- Evaluate initial drafts for errors, gaps that require additional research.
- Provide guidance on better organizing structure and flow.
- Assess tone, voice and brand alignment.

Expert Review:
- Ask medical experts related to article topic to validate accuracy of information.
- Verify advice follows ethical guidelines accepted by the medical community.   
- Request quotes that lend credibility and reinforce key points.

Legal Review:  
- Send drafts discussing new research, controversial topics for legal approval.
- Confirm content meets regulatory standards for health claims and liability risks.
- Address any recommended edits to mitigate brand reputation risk.

Quality Checklist:  
- Scrutinize final draft against PositiveMed's high editorial standards:
- Medical accuracy - error-free facts/statistics, supported claims 
- Logical flow - smooth transitions, complementary sections  
- Reader value - insightful analysis beyond fluffy content
- Brand alignment - uplifting tone, inclusive messaging
- Optimization - keywords in metadata, internal links, page speed
- Strong conclusion - memorable takeaways, relevant next steps/resources for readers
- Proper attribution - all references cited, ownership of reused assets 

Proofreading Process

The final quality safeguard is meticulous proofreading and citation checks.

Grammar & Spelling:
- Use editing tools like Grammarly along with manual review.  
- Check for typos, punctuation errors, misused words, repetitive language.

Formatting:  
- Validate formatting consistency across headlines, bodies, captions etc.  
- Fix spacing/indentation, text styling, text wrapping issues.

Citations:
- Cross-check in-text citations against reference links and sources.
- Ensure citation style matches PositiveMed guidelines.
- Add citations if needed for any statistics or direct quotes.

HTML Proofing:  
- Review article on site mockup to catch issues with text rendering, image sizing etc.
- Confirm formatting displays properly across devices - desktop, mobile, tablet. 

Final Touches:
- Make final pass over article when fresh and focused to spot lingering errors.
- Tweak wording, strengthen transition phrases, vary sentence lengths for polish.  
- Ensure conclusion drives home core message and key takeaways.

This 5,000+ word SOP offers a blueprint for creating well-researched, expert-approved, SEO-optimized articles that engage and inform PositiveMed's audience. Please let me know if you need any clarification or have additional guidelines to incorporate.
"""


draft_blog = llm(DRAFT_AGENT_SYSTEM_PROMPT)
draft_out = print(
    colored(
        f"""
    
    ------------------------------------
    Drafter Writer Agent
    -----------------------------

    Draft:
    ------------------------
    {draft_blog}
    
    """,
        "red",
    )
)


# Agent that reviews the draft
review_agent = llm(get_review_prompt(draft_blog))
reviewed_draft = print(
    colored(
        f"""
    
    ------------------------------------
    Quality Assurance Writer Agent
    -----------------------------

    Complete Narrative:
    ------------------------
    {draft_blog}
    
    """,
        "blue",
    )
)


# Agent that publishes on social media
distribution_agent = llm(
    social_media_prompt(draft_blog, goal="Clicks and engagement")
)
distribution_agent_out = print(
    colored(
        f"""
        --------------------------------
        Distribution Agent
        -------------------

        Social Media Posts
        -------------------
        {distribution_agent}

        """, "magenta"
    )
)