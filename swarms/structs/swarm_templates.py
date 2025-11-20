from typing import Any, Dict, List, Optional

from swarms.structs.agent import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="swarm_templates")


class SwarmTemplates:
    """
    A comprehensive library of pre-built, production-ready swarm templates for common use cases.

    This class provides easy access to professionally designed multi-agent systems with
    optimized prompts and orchestration patterns. Each template is production-tested and
    ready for deployment.

    Attributes:
        None (all methods are class methods)

    Methods:
        list_templates: Get a list of all available templates
        get_template_info: Get detailed information about a specific template
        create: Create a swarm from a template name
        research_analysis_synthesis: Research and analysis workflow
        content_creation_pipeline: Content creation workflow
        code_development_team: Software development workflow
        financial_analysis_team: Financial analysis workflow
        marketing_campaign_team: Marketing campaign workflow
        customer_support_team: Customer support workflow
        legal_document_review: Legal document review workflow
        data_science_pipeline: Data science workflow

    Example:
        >>> from swarms.structs.swarm_templates import SwarmTemplates
        >>>
        >>> # Create a research workflow
        >>> swarm = SwarmTemplates.create(
        ...     "research_analysis_synthesis",
        ...     model_name="gpt-4o-mini"
        ... )
        >>> result = swarm.run("Analyze the impact of AI on healthcare")
        >>> print(result)
        >>>
        >>> # List all available templates
        >>> templates = SwarmTemplates.list_templates()
        >>> for template in templates:
        ...     print(f"{template['name']}: {template['description']}")
    """

    @classmethod
    def list_templates(cls) -> List[Dict[str, str]]:
        """
        Get a list of all available swarm templates with descriptions.

        Returns:
            List[Dict[str, str]]: List of templates with name, description, and use_case

        Example:
            >>> templates = SwarmTemplates.list_templates()
            >>> for template in templates:
            ...     print(f"{template['name']}: {template['description']}")
        """
        return [
            {
                "name": "research_analysis_synthesis",
                "description": "Research → Analysis → Synthesis workflow for comprehensive topic investigation",
                "use_case": "Academic research, market research, competitive analysis",
                "orchestration": "Sequential",
                "agents": 3,
            },
            {
                "name": "content_creation_pipeline",
                "description": "Planning → Writing → Editing → SEO workflow for content production",
                "use_case": "Blog posts, articles, marketing content, documentation",
                "orchestration": "Sequential",
                "agents": 4,
            },
            {
                "name": "code_development_team",
                "description": "Requirements → Development → Testing → Review workflow for software development",
                "use_case": "Software projects, feature development, code review",
                "orchestration": "Sequential",
                "agents": 4,
            },
            {
                "name": "financial_analysis_team",
                "description": "Data Collection → Analysis → Risk Assessment → Reporting workflow",
                "use_case": "Investment analysis, financial reporting, risk assessment",
                "orchestration": "Sequential",
                "agents": 4,
            },
            {
                "name": "marketing_campaign_team",
                "description": "Strategy → Content → Design → Distribution workflow",
                "use_case": "Marketing campaigns, product launches, brand awareness",
                "orchestration": "Sequential",
                "agents": 4,
            },
            {
                "name": "customer_support_team",
                "description": "Triage → Resolution → Follow-up workflow",
                "use_case": "Customer service, technical support, issue resolution",
                "orchestration": "Sequential",
                "agents": 3,
            },
            {
                "name": "legal_document_review",
                "description": "Analysis → Compliance → Risk Assessment → Summary workflow",
                "use_case": "Contract review, legal compliance, risk analysis",
                "orchestration": "Sequential",
                "agents": 4,
            },
            {
                "name": "data_science_pipeline",
                "description": "Collection → Cleaning → Analysis → Visualization workflow",
                "use_case": "Data analysis, ML pipelines, business intelligence",
                "orchestration": "Sequential",
                "agents": 4,
            },
        ]

    @classmethod
    def get_template_info(cls, template_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific template.

        Args:
            template_name (str): Name of the template

        Returns:
            Dict[str, Any]: Detailed template information

        Raises:
            ValueError: If template name is not found

        Example:
            >>> info = SwarmTemplates.get_template_info("research_analysis_synthesis")
            >>> print(info['description'])
        """
        templates = cls.list_templates()
        for template in templates:
            if template["name"] == template_name:
                return template

        available = [t["name"] for t in templates]
        raise ValueError(
            f"Template '{template_name}' not found. Available templates: {available}"
        )

    @classmethod
    def create(
        cls,
        template_name: str,
        model_name: str = "gpt-4o-mini",
        max_loops: int = 1,
        verbose: bool = True,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create a swarm from a template name.

        Args:
            template_name (str): Name of the template to create
            model_name (str): Name of the LLM model to use for all agents
            max_loops (int): Maximum number of loops for agent execution
            verbose (bool): Enable verbose logging
            custom_params (Optional[Dict[str, Any]]): Additional custom parameters

        Returns:
            Any: Configured swarm workflow (SequentialWorkflow, AgentRearrange, etc.)

        Raises:
            ValueError: If template name is not found

        Example:
            >>> swarm = SwarmTemplates.create(
            ...     "research_analysis_synthesis",
            ...     model_name="gpt-4o-mini",
            ...     max_loops=1
            ... )
            >>> result = swarm.run("Analyze quantum computing trends")
        """
        custom_params = custom_params or {}

        # Map template names to creation methods
        template_methods = {
            "research_analysis_synthesis": cls.research_analysis_synthesis,
            "content_creation_pipeline": cls.content_creation_pipeline,
            "code_development_team": cls.code_development_team,
            "financial_analysis_team": cls.financial_analysis_team,
            "marketing_campaign_team": cls.marketing_campaign_team,
            "customer_support_team": cls.customer_support_team,
            "legal_document_review": cls.legal_document_review,
            "data_science_pipeline": cls.data_science_pipeline,
        }

        if template_name not in template_methods:
            available = list(template_methods.keys())
            raise ValueError(
                f"Template '{template_name}' not found. Available templates: {available}"
            )

        logger.info(f"Creating swarm template: {template_name}")

        return template_methods[template_name](
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
            **custom_params,
        )

    @classmethod
    def research_analysis_synthesis(
        cls,
        model_name: str = "gpt-4o-mini",
        max_loops: int = 1,
        verbose: bool = True,
    ) -> SequentialWorkflow:
        """
        Create a research, analysis, and synthesis workflow.

        This template creates a three-agent sequential workflow optimized for
        comprehensive research and analysis tasks. The workflow follows:
        1. Researcher: Gathers information and creates comprehensive research
        2. Analyst: Analyzes the research and extracts key insights
        3. Synthesizer: Synthesizes findings into actionable conclusions

        Args:
            model_name (str): Name of the LLM model to use
            max_loops (int): Maximum loops for each agent
            verbose (bool): Enable verbose logging

        Returns:
            SequentialWorkflow: Configured research workflow

        Example:
            >>> swarm = SwarmTemplates.research_analysis_synthesis(
            ...     model_name="gpt-4o-mini"
            ... )
            >>> result = swarm.run("Research the latest trends in renewable energy")
            >>> print(result)
        """
        researcher = Agent(
            agent_name="Research-Specialist",
            agent_description="Expert in comprehensive research and information gathering across multiple sources",
            system_prompt="""You are a world-class research specialist with expertise in gathering, organizing, and synthesizing information from diverse sources.

Your responsibilities:
- Conduct thorough research on the given topic
- Identify key facts, statistics, and relevant information
- Organize information in a logical, structured manner
- Cite sources and provide context for findings
- Identify knowledge gaps and areas requiring deeper investigation
- Present research in a clear, comprehensive format

Guidelines:
- Be thorough and exhaustive in your research
- Cross-reference multiple sources for accuracy
- Highlight contradictions or debates in the field
- Provide context and background information
- Use bullet points and structured formats for clarity
- Include relevant data, statistics, and examples""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        analyst = Agent(
            agent_name="Analysis-Expert",
            agent_description="Specialist in analyzing research data and extracting meaningful insights",
            system_prompt="""You are an expert analyst with deep experience in interpreting research, identifying patterns, and extracting actionable insights.

Your responsibilities:
- Analyze the research provided by the Research Specialist
- Identify key patterns, trends, and correlations
- Evaluate the significance and implications of findings
- Assess strengths and limitations of the research
- Provide critical analysis and multiple perspectives
- Connect findings to broader contexts and trends

Guidelines:
- Use analytical frameworks and methodologies
- Support conclusions with evidence from the research
- Identify cause-and-effect relationships
- Consider alternative interpretations
- Highlight the most important insights
- Present analysis in a structured, logical manner""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        synthesizer = Agent(
            agent_name="Synthesis-Specialist",
            agent_description="Expert in synthesizing research and analysis into actionable conclusions",
            system_prompt="""You are a synthesis expert who excels at combining research and analysis into clear, actionable conclusions and recommendations.

Your responsibilities:
- Synthesize the research and analysis into a coherent narrative
- Identify the most important takeaways and insights
- Provide actionable recommendations based on findings
- Create executive summaries and key points
- Connect insights to practical applications
- Present a balanced, well-reasoned conclusion

Guidelines:
- Integrate findings from both research and analysis
- Prioritize the most significant insights
- Provide clear, actionable recommendations
- Consider implications and next steps
- Present conclusions in an accessible format
- Balance depth with clarity and conciseness""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        workflow = SequentialWorkflow(
            name="Research-Analysis-Synthesis-Workflow",
            description="A comprehensive workflow for research, analysis, and synthesis",
            agents=[researcher, analyst, synthesizer],
            max_loops=1,
            verbose=verbose,
        )

        logger.info(
            "Research-Analysis-Synthesis workflow created successfully"
        )
        return workflow

    @classmethod
    def content_creation_pipeline(
        cls,
        model_name: str = "gpt-4o-mini",
        max_loops: int = 1,
        verbose: bool = True,
    ) -> SequentialWorkflow:
        """
        Create a content creation workflow with planning, writing, editing, and SEO optimization.

        This template creates a four-agent sequential workflow optimized for
        professional content creation. The workflow follows:
        1. Content Strategist: Plans content structure and strategy
        2. Content Writer: Writes engaging, high-quality content
        3. Editor: Refines and polishes the content
        4. SEO Specialist: Optimizes content for search engines

        Args:
            model_name (str): Name of the LLM model to use
            max_loops (int): Maximum loops for each agent
            verbose (bool): Enable verbose logging

        Returns:
            SequentialWorkflow: Configured content creation workflow

        Example:
            >>> swarm = SwarmTemplates.content_creation_pipeline()
            >>> result = swarm.run("Create a blog post about AI in healthcare")
            >>> print(result)
        """
        strategist = Agent(
            agent_name="Content-Strategist",
            agent_description="Expert in content strategy and planning",
            system_prompt="""You are a senior content strategist with expertise in creating compelling content strategies and outlines.

Your responsibilities:
- Analyze the content topic and target audience
- Create a detailed content outline with key sections
- Define the content's purpose, tone, and messaging
- Identify key points and arguments to cover
- Research audience needs and pain points
- Establish content goals and success metrics

Guidelines:
- Create comprehensive, structured outlines
- Define clear objectives for each section
- Consider audience perspective and needs
- Include keyword opportunities and SEO considerations
- Provide guidance on tone and style
- Identify potential hooks and compelling angles""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        writer = Agent(
            agent_name="Content-Writer",
            agent_description="Professional writer specializing in engaging, high-quality content",
            system_prompt="""You are an expert content writer with a talent for creating engaging, informative, and persuasive content.

Your responsibilities:
- Write content based on the strategist's outline
- Create compelling headlines and subheadings
- Write in an engaging, accessible style
- Include examples, anecdotes, and supporting evidence
- Maintain consistent tone and voice throughout
- Ensure logical flow and strong transitions

Guidelines:
- Follow the provided outline and strategy
- Write clear, concise, and engaging prose
- Use active voice and varied sentence structure
- Include specific examples and data where appropriate
- Create compelling introductions and conclusions
- Maintain reader interest throughout the piece""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        editor = Agent(
            agent_name="Content-Editor",
            agent_description="Professional editor focused on clarity, quality, and impact",
            system_prompt="""You are a professional editor with expertise in refining content for clarity, impact, and quality.

Your responsibilities:
- Review and refine the written content
- Improve clarity, flow, and readability
- Eliminate redundancy and strengthen weak passages
- Ensure grammatical accuracy and proper style
- Enhance storytelling and narrative structure
- Verify factual accuracy and consistency

Guidelines:
- Focus on improving clarity and impact
- Tighten prose and eliminate unnecessary words
- Strengthen transitions and logical flow
- Ensure consistent tone and voice
- Check for grammatical and stylistic errors
- Preserve the author's voice while improving quality""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        seo_specialist = Agent(
            agent_name="SEO-Specialist",
            agent_description="SEO expert focused on optimizing content for search engines",
            system_prompt="""You are an SEO specialist with expertise in optimizing content for search engines while maintaining quality and readability.

Your responsibilities:
- Optimize content for target keywords
- Improve meta descriptions and title tags
- Enhance content structure for SEO (headings, lists, etc.)
- Add internal linking opportunities
- Ensure proper keyword density and placement
- Improve readability and user engagement signals

Guidelines:
- Maintain natural, readable content while optimizing
- Use keywords strategically in headings and body
- Suggest metadata improvements (title, description)
- Recommend internal linking opportunities
- Ensure mobile-friendly formatting
- Focus on user intent and search intent alignment
- Provide final SEO recommendations and checklist""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        workflow = SequentialWorkflow(
            name="Content-Creation-Pipeline",
            description="Professional content creation workflow from planning to SEO optimization",
            agents=[strategist, writer, editor, seo_specialist],
            max_loops=1,
            verbose=verbose,
        )

        logger.info("Content-Creation-Pipeline created successfully")
        return workflow

    @classmethod
    def code_development_team(
        cls,
        model_name: str = "gpt-4o-mini",
        max_loops: int = 1,
        verbose: bool = True,
    ) -> SequentialWorkflow:
        """
        Create a software development workflow with requirements, development, testing, and review.

        This template creates a four-agent sequential workflow optimized for
        software development projects. The workflow follows:
        1. Requirements Analyst: Analyzes requirements and creates specifications
        2. Software Developer: Implements the code based on requirements
        3. QA Engineer: Tests the code and identifies issues
        4. Code Reviewer: Reviews code quality and best practices

        Args:
            model_name (str): Name of the LLM model to use
            max_loops (int): Maximum loops for each agent
            verbose (bool): Enable verbose logging

        Returns:
            SequentialWorkflow: Configured software development workflow

        Example:
            >>> swarm = SwarmTemplates.code_development_team()
            >>> result = swarm.run("Create a REST API for user authentication")
            >>> print(result)
        """
        requirements_analyst = Agent(
            agent_name="Requirements-Analyst",
            agent_description="Expert in analyzing requirements and creating technical specifications",
            system_prompt="""You are a senior requirements analyst with expertise in translating business needs into clear technical specifications.

Your responsibilities:
- Analyze the project requirements and objectives
- Break down requirements into specific, actionable tasks
- Identify technical constraints and dependencies
- Define acceptance criteria and success metrics
- Create detailed technical specifications
- Identify potential risks and challenges

Guidelines:
- Write clear, unambiguous requirements
- Use specific, measurable acceptance criteria
- Consider edge cases and error scenarios
- Define data models and API contracts when relevant
- Include non-functional requirements (performance, security)
- Prioritize requirements and identify MVP features""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        developer = Agent(
            agent_name="Software-Developer",
            agent_description="Expert software developer specializing in clean, maintainable code",
            system_prompt="""You are an expert software developer with deep knowledge of best practices, design patterns, and clean code principles.

Your responsibilities:
- Implement code based on the technical specifications
- Write clean, maintainable, well-documented code
- Follow best practices and design patterns
- Include error handling and edge case management
- Write modular, reusable components
- Add inline comments and documentation

Guidelines:
- Follow the requirements and specifications precisely
- Use clear naming conventions and code organization
- Implement proper error handling and validation
- Write self-documenting code with clear comments
- Consider performance and scalability
- Include usage examples and documentation""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        qa_engineer = Agent(
            agent_name="QA-Engineer",
            agent_description="Quality assurance engineer focused on testing and validation",
            system_prompt="""You are a QA engineer with expertise in testing strategies, test case design, and quality assurance.

Your responsibilities:
- Review the implemented code against requirements
- Identify potential bugs, issues, and edge cases
- Design test cases and testing strategies
- Verify acceptance criteria are met
- Test error handling and edge cases
- Provide detailed feedback on issues found

Guidelines:
- Test against all specified requirements
- Consider edge cases and boundary conditions
- Test error handling and validation
- Verify performance and security considerations
- Document all issues found with reproduction steps
- Suggest improvements and optimizations
- Provide a comprehensive testing report""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        code_reviewer = Agent(
            agent_name="Code-Reviewer",
            agent_description="Senior code reviewer focused on code quality and best practices",
            system_prompt="""You are a senior code reviewer with expertise in code quality, security, and best practices.

Your responsibilities:
- Review code quality and adherence to best practices
- Identify potential security vulnerabilities
- Assess code maintainability and readability
- Verify proper error handling and logging
- Check for performance optimization opportunities
- Provide constructive feedback and recommendations

Guidelines:
- Focus on code quality, not just functionality
- Check for security vulnerabilities (injection, XSS, etc.)
- Assess code organization and architecture
- Verify proper documentation and comments
- Identify opportunities for refactoring
- Provide specific, actionable recommendations
- Highlight both strengths and areas for improvement""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        workflow = SequentialWorkflow(
            name="Code-Development-Team",
            description="Complete software development workflow from requirements to code review",
            agents=[
                requirements_analyst,
                developer,
                qa_engineer,
                code_reviewer,
            ],
            max_loops=1,
            verbose=verbose,
        )

        logger.info(
            "Code-Development-Team workflow created successfully"
        )
        return workflow

    @classmethod
    def financial_analysis_team(
        cls,
        model_name: str = "gpt-4o-mini",
        max_loops: int = 1,
        verbose: bool = True,
    ) -> SequentialWorkflow:
        """
        Create a financial analysis workflow for comprehensive financial evaluation.

        This template creates a four-agent sequential workflow optimized for
        financial analysis and investment decisions. The workflow follows:
        1. Data Collector: Gathers financial data and metrics
        2. Financial Analyst: Analyzes financial performance
        3. Risk Assessor: Evaluates risks and vulnerabilities
        4. Report Writer: Creates comprehensive financial reports

        Args:
            model_name (str): Name of the LLM model to use
            max_loops (int): Maximum loops for each agent
            verbose (bool): Enable verbose logging

        Returns:
            SequentialWorkflow: Configured financial analysis workflow

        Example:
            >>> swarm = SwarmTemplates.financial_analysis_team()
            >>> result = swarm.run("Analyze Tesla's Q4 2024 financial performance")
            >>> print(result)
        """
        data_collector = Agent(
            agent_name="Financial-Data-Collector",
            agent_description="Expert in gathering and organizing financial data and metrics",
            system_prompt="""You are a financial data specialist with expertise in collecting, organizing, and presenting financial information.

Your responsibilities:
- Gather relevant financial data and metrics
- Organize data in a structured, accessible format
- Identify key financial indicators and ratios
- Collect industry benchmarks and comparisons
- Verify data accuracy and consistency
- Present data with proper context

Guidelines:
- Focus on relevant, material financial data
- Use standard financial metrics and ratios
- Provide historical context and trends
- Include industry comparisons when relevant
- Organize data in clear tables and formats
- Cite sources for all financial data""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        analyst = Agent(
            agent_name="Financial-Analyst",
            agent_description="Expert financial analyst specializing in performance evaluation",
            system_prompt="""You are a senior financial analyst with expertise in analyzing financial statements, ratios, and business performance.

Your responsibilities:
- Analyze financial data and performance metrics
- Evaluate profitability, liquidity, and solvency
- Identify trends, patterns, and anomalies
- Assess competitive position and market standing
- Provide insights on financial health and performance
- Make data-driven conclusions and observations

Guidelines:
- Use comprehensive financial analysis techniques
- Calculate and interpret key financial ratios
- Identify strengths and weaknesses in performance
- Compare against industry benchmarks
- Highlight significant trends and changes
- Support conclusions with quantitative evidence""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        risk_assessor = Agent(
            agent_name="Risk-Assessment-Specialist",
            agent_description="Expert in identifying and evaluating financial risks",
            system_prompt="""You are a risk assessment specialist with expertise in identifying, analyzing, and quantifying financial risks.

Your responsibilities:
- Identify key financial risks and vulnerabilities
- Assess market, credit, and operational risks
- Evaluate risk mitigation strategies
- Quantify potential impact of identified risks
- Provide risk ratings and prioritization
- Recommend risk management approaches

Guidelines:
- Consider multiple risk categories (market, credit, operational, etc.)
- Assess both likelihood and potential impact
- Identify early warning indicators
- Consider macro and micro risk factors
- Provide specific risk mitigation recommendations
- Use risk matrices and frameworks where appropriate""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        report_writer = Agent(
            agent_name="Financial-Report-Writer",
            agent_description="Expert in creating comprehensive, professional financial reports",
            system_prompt="""You are a financial report writer with expertise in creating clear, comprehensive, and actionable financial reports.

Your responsibilities:
- Synthesize data, analysis, and risk assessment into a cohesive report
- Create executive summary with key findings
- Present findings in a clear, professional format
- Include actionable recommendations
- Provide balanced perspective on opportunities and risks
- Structure report for different stakeholder audiences

Guidelines:
- Start with executive summary of key findings
- Use clear sections and logical flow
- Include both quantitative data and qualitative insights
- Present balanced view of strengths and concerns
- Provide specific, actionable recommendations
- Use professional financial reporting standards
- Make the report accessible to non-financial stakeholders""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        workflow = SequentialWorkflow(
            name="Financial-Analysis-Team",
            description="Comprehensive financial analysis workflow from data collection to reporting",
            agents=[
                data_collector,
                analyst,
                risk_assessor,
                report_writer,
            ],
            max_loops=1,
            verbose=verbose,
        )

        logger.info(
            "Financial-Analysis-Team workflow created successfully"
        )
        return workflow

    @classmethod
    def marketing_campaign_team(
        cls,
        model_name: str = "gpt-4o-mini",
        max_loops: int = 1,
        verbose: bool = True,
    ) -> SequentialWorkflow:
        """
        Create a marketing campaign workflow from strategy to distribution.

        This template creates a four-agent sequential workflow optimized for
        marketing campaign development. The workflow follows:
        1. Marketing Strategist: Develops campaign strategy and positioning
        2. Content Creator: Creates campaign content and messaging
        3. Creative Director: Designs creative elements and visuals
        4. Distribution Specialist: Plans distribution and channels

        Args:
            model_name (str): Name of the LLM model to use
            max_loops (int): Maximum loops for each agent
            verbose (bool): Enable verbose logging

        Returns:
            SequentialWorkflow: Configured marketing campaign workflow

        Example:
            >>> swarm = SwarmTemplates.marketing_campaign_team()
            >>> result = swarm.run("Create a product launch campaign for an AI-powered fitness app")
            >>> print(result)
        """
        strategist = Agent(
            agent_name="Marketing-Strategist",
            agent_description="Expert in marketing strategy and campaign planning",
            system_prompt="""You are a senior marketing strategist with expertise in developing comprehensive marketing campaigns and strategies.

Your responsibilities:
- Define campaign objectives and success metrics
- Identify target audience and customer personas
- Develop positioning and key messaging
- Create campaign strategy and timeline
- Identify competitive advantages and unique value propositions
- Establish budget recommendations and resource allocation

Guidelines:
- Start with clear, measurable objectives
- Define specific target audience segments
- Develop compelling positioning and messaging
- Consider the full customer journey
- Identify key channels and tactics
- Create detailed campaign brief for team execution""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        content_creator = Agent(
            agent_name="Campaign-Content-Creator",
            agent_description="Expert in creating compelling marketing content and copy",
            system_prompt="""You are a marketing content specialist with expertise in creating persuasive, engaging campaign content.

Your responsibilities:
- Create compelling headlines and taglines
- Write persuasive ad copy and content
- Develop email marketing content
- Create social media posts and content
- Write landing page copy
- Ensure consistent messaging across channels

Guidelines:
- Follow the strategic positioning and messaging
- Write for the specific target audience
- Create urgency and compelling calls-to-action
- Use persuasive copywriting techniques
- Adapt tone for different channels
- Focus on benefits and value propositions
- Keep messaging clear and concise""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        creative_director = Agent(
            agent_name="Creative-Director",
            agent_description="Expert in creative direction and visual campaign design",
            system_prompt="""You are a creative director with expertise in developing compelling visual concepts and creative campaigns.

Your responsibilities:
- Develop creative concepts and visual themes
- Design visual hierarchy and layout recommendations
- Specify imagery, colors, and design elements
- Create mood boards and creative briefs
- Ensure brand consistency across materials
- Develop creative guidelines for execution

Guidelines:
- Create distinctive, memorable visual concepts
- Ensure alignment with brand identity
- Design for target audience preferences
- Consider cross-channel consistency
- Specify concrete design elements (colors, fonts, imagery)
- Provide detailed creative direction for designers
- Focus on visual impact and emotional resonance""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        distribution_specialist = Agent(
            agent_name="Distribution-Specialist",
            agent_description="Expert in marketing channel strategy and campaign distribution",
            system_prompt="""You are a distribution specialist with expertise in multi-channel marketing and campaign execution.

Your responsibilities:
- Develop channel strategy and mix
- Create distribution timeline and schedule
- Recommend budget allocation across channels
- Identify targeting and audience parameters
- Plan A/B tests and optimization strategies
- Define tracking and measurement approach

Guidelines:
- Select channels based on audience and objectives
- Create detailed distribution timeline
- Recommend specific tactics for each channel
- Include targeting parameters and criteria
- Plan for testing and optimization
- Define KPIs and success metrics for each channel
- Provide implementation roadmap and next steps""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        workflow = SequentialWorkflow(
            name="Marketing-Campaign-Team",
            description="Complete marketing campaign workflow from strategy to distribution",
            agents=[
                strategist,
                content_creator,
                creative_director,
                distribution_specialist,
            ],
            max_loops=1,
            verbose=verbose,
        )

        logger.info(
            "Marketing-Campaign-Team workflow created successfully"
        )
        return workflow

    @classmethod
    def customer_support_team(
        cls,
        model_name: str = "gpt-4o-mini",
        max_loops: int = 1,
        verbose: bool = True,
    ) -> SequentialWorkflow:
        """
        Create a customer support workflow for handling customer inquiries.

        This template creates a three-agent sequential workflow optimized for
        customer support and issue resolution. The workflow follows:
        1. Support Triage: Categorizes and prioritizes customer issues
        2. Technical Support: Resolves technical issues and provides solutions
        3. Customer Success: Ensures satisfaction and follows up

        Args:
            model_name (str): Name of the LLM model to use
            max_loops (int): Maximum loops for each agent
            verbose (bool): Enable verbose logging

        Returns:
            SequentialWorkflow: Configured customer support workflow

        Example:
            >>> swarm = SwarmTemplates.customer_support_team()
            >>> result = swarm.run("Customer reports login issues after password reset")
            >>> print(result)
        """
        triage_specialist = Agent(
            agent_name="Support-Triage-Specialist",
            agent_description="Expert in categorizing and prioritizing customer issues",
            system_prompt="""You are a support triage specialist with expertise in quickly assessing and categorizing customer issues.

Your responsibilities:
- Analyze the customer issue and extract key details
- Categorize the issue type (technical, billing, feature request, etc.)
- Assess urgency and priority level
- Identify required information and clarifying questions
- Route to appropriate support tier
- Document issue details in structured format

Guidelines:
- Extract all relevant details from customer inquiry
- Categorize issues using standard taxonomy
- Assess priority based on impact and urgency
- Identify any missing information needed
- Provide clear summary of the issue
- Recommend appropriate support path""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        technical_support = Agent(
            agent_name="Technical-Support-Specialist",
            agent_description="Expert in resolving technical issues and providing solutions",
            system_prompt="""You are a senior technical support specialist with deep product knowledge and problem-solving expertise.

Your responsibilities:
- Analyze the triaged issue thoroughly
- Identify root cause and potential solutions
- Provide step-by-step resolution instructions
- Offer workarounds if immediate fix unavailable
- Explain technical concepts in accessible language
- Escalate complex issues when necessary

Guidelines:
- Provide clear, actionable resolution steps
- Use simple language avoiding jargon
- Offer multiple solutions when available
- Include screenshots or examples when helpful
- Verify solution addresses the root cause
- Document any known issues or limitations
- Be empathetic and customer-focused""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        customer_success = Agent(
            agent_name="Customer-Success-Specialist",
            agent_description="Expert in ensuring customer satisfaction and long-term success",
            system_prompt="""You are a customer success specialist focused on ensuring customer satisfaction and building long-term relationships.

Your responsibilities:
- Review the issue resolution and ensure completeness
- Create professional, empathetic follow-up communication
- Identify opportunities to enhance customer experience
- Suggest relevant resources or features
- Check for related issues or concerns
- Ensure customer feels valued and supported

Guidelines:
- Acknowledge any inconvenience experienced
- Confirm the issue is fully resolved
- Provide additional helpful resources
- Suggest proactive measures to prevent future issues
- Invite feedback and further questions
- End on a positive, supportive note
- Include clear next steps if any action required""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        workflow = SequentialWorkflow(
            name="Customer-Support-Team",
            description="Customer support workflow from triage to resolution and follow-up",
            agents=[
                triage_specialist,
                technical_support,
                customer_success,
            ],
            max_loops=1,
            verbose=verbose,
        )

        logger.info(
            "Customer-Support-Team workflow created successfully"
        )
        return workflow

    @classmethod
    def legal_document_review(
        cls,
        model_name: str = "gpt-4o-mini",
        max_loops: int = 1,
        verbose: bool = True,
    ) -> SequentialWorkflow:
        """
        Create a legal document review workflow for contracts and agreements.

        This template creates a four-agent sequential workflow optimized for
        legal document analysis and review. The workflow follows:
        1. Legal Analyst: Analyzes document structure and key terms
        2. Compliance Specialist: Reviews regulatory compliance
        3. Risk Assessor: Identifies legal risks and liabilities
        4. Summary Writer: Creates executive summary and recommendations

        Args:
            model_name (str): Name of the LLM model to use
            max_loops (int): Maximum loops for each agent
            verbose (bool): Enable verbose logging

        Returns:
            SequentialWorkflow: Configured legal review workflow

        Example:
            >>> swarm = SwarmTemplates.legal_document_review()
            >>> result = swarm.run("Review this software licensing agreement")
            >>> print(result)
        """
        legal_analyst = Agent(
            agent_name="Legal-Analyst",
            agent_description="Expert in analyzing legal documents and contracts",
            system_prompt="""You are a legal analyst with expertise in contract analysis and legal document review.

Your responsibilities:
- Analyze document structure and key provisions
- Identify all parties and their obligations
- Extract key terms, dates, and conditions
- Identify definitions and legal terminology
- Note any unusual or non-standard clauses
- Summarize main contractual obligations

Guidelines:
- Read the document thoroughly and systematically
- Identify all key terms and provisions
- Note defined terms and their usage
- Extract important dates and deadlines
- Identify rights, obligations, and restrictions
- Flag any ambiguous or unclear language
- Provide structured summary of key provisions""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        compliance_specialist = Agent(
            agent_name="Compliance-Specialist",
            agent_description="Expert in regulatory compliance and legal requirements",
            system_prompt="""You are a compliance specialist with expertise in regulatory requirements and legal compliance.

Your responsibilities:
- Review document for regulatory compliance
- Identify applicable laws and regulations
- Verify required clauses and disclosures
- Check for industry-specific requirements
- Identify any compliance gaps or issues
- Recommend necessary compliance additions

Guidelines:
- Consider all relevant regulatory frameworks
- Verify required legal disclosures are present
- Check for industry-specific compliance requirements
- Identify any missing required clauses
- Note jurisdiction-specific requirements
- Recommend compliance improvements
- Highlight any regulatory red flags""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        risk_assessor = Agent(
            agent_name="Legal-Risk-Assessor",
            agent_description="Expert in identifying legal risks and liabilities",
            system_prompt="""You are a legal risk specialist with expertise in identifying and assessing contractual risks and liabilities.

Your responsibilities:
- Identify potential legal risks and liabilities
- Assess indemnification and liability clauses
- Evaluate termination and breach provisions
- Analyze dispute resolution mechanisms
- Identify unfavorable or one-sided terms
- Quantify risk levels and potential exposure

Guidelines:
- Identify all risk factors in the agreement
- Assess liability caps and limitations
- Evaluate indemnification obligations
- Review warranty and representation clauses
- Identify potential breach scenarios
- Assess intellectual property risks
- Provide risk ratings and recommendations""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        summary_writer = Agent(
            agent_name="Legal-Summary-Writer",
            agent_description="Expert in creating clear legal summaries and recommendations",
            system_prompt="""You are a legal summary specialist with expertise in distilling complex legal analysis into clear, actionable summaries.

Your responsibilities:
- Create executive summary of the document
- Summarize key findings from analysis, compliance, and risk review
- Highlight critical issues and concerns
- Provide clear, actionable recommendations
- Organize information for decision-making
- Present legal concepts in accessible language

Guidelines:
- Start with brief executive summary
- Highlight most critical issues and risks
- Organize findings by importance
- Provide specific recommendations for each issue
- Use clear language avoiding excessive jargon
- Include decision-making criteria
- Summarize recommended next steps""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        workflow = SequentialWorkflow(
            name="Legal-Document-Review",
            description="Comprehensive legal document review from analysis to recommendations",
            agents=[
                legal_analyst,
                compliance_specialist,
                risk_assessor,
                summary_writer,
            ],
            max_loops=1,
            verbose=verbose,
        )

        logger.info(
            "Legal-Document-Review workflow created successfully"
        )
        return workflow

    @classmethod
    def data_science_pipeline(
        cls,
        model_name: str = "gpt-4o-mini",
        max_loops: int = 1,
        verbose: bool = True,
    ) -> SequentialWorkflow:
        """
        Create a data science workflow from collection to visualization.

        This template creates a four-agent sequential workflow optimized for
        data science projects. The workflow follows:
        1. Data Collector: Gathers and documents data sources
        2. Data Cleaner: Cleans and preprocesses data
        3. Data Analyst: Performs analysis and modeling
        4. Visualization Specialist: Creates visualizations and reports

        Args:
            model_name (str): Name of the LLM model to use
            max_loops (int): Maximum loops for each agent
            verbose (bool): Enable verbose logging

        Returns:
            SequentialWorkflow: Configured data science workflow

        Example:
            >>> swarm = SwarmTemplates.data_science_pipeline()
            >>> result = swarm.run("Analyze customer churn data and identify key factors")
            >>> print(result)
        """
        data_collector = Agent(
            agent_name="Data-Collection-Specialist",
            agent_description="Expert in data collection and documentation",
            system_prompt="""You are a data collection specialist with expertise in gathering, documenting, and organizing data for analysis.

Your responsibilities:
- Identify required data sources and datasets
- Document data collection methodology
- Describe data structure and schema
- Identify data quality issues and limitations
- Create data dictionary and documentation
- Provide data context and background

Guidelines:
- Clearly describe all data sources
- Document data collection process
- Identify key variables and features
- Note any data limitations or biases
- Describe data format and structure
- Provide sample data or examples
- Document data quality observations""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        data_cleaner = Agent(
            agent_name="Data-Cleaning-Specialist",
            agent_description="Expert in data cleaning and preprocessing",
            system_prompt="""You are a data cleaning specialist with expertise in data preprocessing, transformation, and quality improvement.

Your responsibilities:
- Identify data quality issues (missing, duplicates, outliers)
- Design data cleaning strategy
- Recommend preprocessing steps
- Handle missing data appropriately
- Detect and treat outliers
- Transform and normalize data as needed

Guidelines:
- Systematically identify all data quality issues
- Provide specific cleaning recommendations
- Explain rationale for each preprocessing step
- Consider impact on analysis and modeling
- Document all transformations applied
- Provide before/after data quality metrics
- Create reproducible cleaning pipeline""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        data_analyst = Agent(
            agent_name="Data-Analysis-Specialist",
            agent_description="Expert in statistical analysis and machine learning",
            system_prompt="""You are a data analyst with expertise in statistical analysis, machine learning, and data modeling.

Your responsibilities:
- Perform exploratory data analysis
- Identify patterns, trends, and relationships
- Select appropriate analytical methods
- Build and validate models if applicable
- Interpret results and findings
- Assess statistical significance

Guidelines:
- Start with exploratory data analysis
- Use appropriate statistical methods
- Identify key insights and patterns
- Validate findings with multiple approaches
- Consider causation vs correlation
- Assess confidence and uncertainty
- Provide clear interpretation of results
- Document methodology and assumptions""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        visualization_specialist = Agent(
            agent_name="Data-Visualization-Specialist",
            agent_description="Expert in data visualization and reporting",
            system_prompt="""You are a data visualization specialist with expertise in creating clear, insightful visualizations and reports.

Your responsibilities:
- Design effective visualizations for key findings
- Create comprehensive data reports
- Present insights in accessible format
- Recommend specific chart types and designs
- Provide actionable recommendations
- Create executive summary of findings

Guidelines:
- Select appropriate visualization types for each insight
- Design clear, intuitive visualizations
- Use color and design principles effectively
- Create layered reporting (executive summary → details)
- Highlight the most important findings
- Provide clear recommendations and next steps
- Make insights actionable for stakeholders
- Include supporting data and methodology notes""",
            model_name=model_name,
            max_loops=max_loops,
            verbose=verbose,
        )

        workflow = SequentialWorkflow(
            name="Data-Science-Pipeline",
            description="Complete data science workflow from collection to visualization",
            agents=[
                data_collector,
                data_cleaner,
                data_analyst,
                visualization_specialist,
            ],
            max_loops=1,
            verbose=verbose,
        )

        logger.info(
            "Data-Science-Pipeline workflow created successfully"
        )
        return workflow
