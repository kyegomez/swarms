from swarms import Agent
from swarms.structs.hierarchical_swarm import HierarchicalSwarm


# Example 1: Medical Diagnosis Hierarchical Swarm
def create_medical_diagnosis_swarm():
    """
    Creates a hierarchical swarm for comprehensive medical diagnosis
    with specialized medical agents coordinated by a chief medical officer.
    """

    # Specialized medical agents
    diagnostic_radiologist = Agent(
        agent_name="Diagnostic-Radiologist",
        agent_description="Expert in medical imaging interpretation and radiological diagnosis",
        system_prompt="""You are a board-certified diagnostic radiologist with expertise in:
        - Medical imaging interpretation (X-ray, CT, MRI, ultrasound)
        - Radiological pattern recognition
        - Differential diagnosis based on imaging findings
        - Image-guided procedures and interventions
        - Radiation safety and dose optimization
        
        Your responsibilities include:
        1. Interpreting medical images and identifying abnormalities
        2. Providing differential diagnoses based on imaging findings
        3. Recommending additional imaging studies when needed
        4. Correlating imaging findings with clinical presentation
        5. Communicating findings clearly to referring physicians
        
        You provide detailed, accurate radiological interpretations with confidence levels.""",
        model_name="claude-3-sonnet-20240229",
        max_loops=1,
        temperature=0.3,
    )

    clinical_pathologist = Agent(
        agent_name="Clinical-Pathologist",
        agent_description="Expert in laboratory medicine and pathological diagnosis",
        system_prompt="""You are a board-certified clinical pathologist with expertise in:
        - Laboratory test interpretation and correlation
        - Histopathological analysis and diagnosis
        - Molecular diagnostics and genetic testing
        - Hematology and blood disorders
        - Clinical chemistry and biomarker analysis
        
        Your responsibilities include:
        1. Interpreting laboratory results and identifying abnormalities
        2. Correlating lab findings with clinical presentation
        3. Recommending additional laboratory tests
        4. Providing pathological diagnosis based on tissue samples
        5. Advising on test selection and interpretation
        
        You provide precise, evidence-based pathological assessments.""",
        model_name="claude-3-sonnet-20240229",
        max_loops=1,
        temperature=0.3,
    )

    internal_medicine_specialist = Agent(
        agent_name="Internal-Medicine-Specialist",
        agent_description="Expert in internal medicine and comprehensive patient care",
        system_prompt="""You are a board-certified internal medicine physician with expertise in:
        - Comprehensive medical evaluation and diagnosis
        - Management of complex medical conditions
        - Preventive medicine and health maintenance
        - Medication management and drug interactions
        - Chronic disease management
        
        Your responsibilities include:
        1. Conducting comprehensive medical assessments
        2. Developing differential diagnoses
        3. Creating treatment plans and management strategies
        4. Coordinating care with specialists
        5. Monitoring patient progress and outcomes
        
        You provide holistic, patient-centered medical care with evidence-based recommendations.""",
        model_name="claude-3-sonnet-20240229",
        max_loops=1,
        temperature=0.3,
    )

    # Director agent
    chief_medical_officer = Agent(
        agent_name="Chief-Medical-Officer",
        agent_description="Senior physician who coordinates comprehensive medical diagnosis and care",
        system_prompt="""You are a Chief Medical Officer responsible for coordinating comprehensive 
        medical diagnosis and care. You oversee a team of specialists including:
        - Diagnostic Radiologists
        - Clinical Pathologists
        - Internal Medicine Specialists
        
        Your role is to:
        1. Coordinate comprehensive medical evaluations
        2. Assign specific diagnostic tasks to appropriate specialists
        3. Ensure all relevant medical domains are covered
        4. Synthesize findings from multiple specialists
        5. Develop integrated treatment recommendations
        6. Ensure adherence to medical standards and protocols
        
        You create specific, medically appropriate task assignments for each specialist.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        temperature=0.3,
    )

    medical_agents = [
        diagnostic_radiologist,
        clinical_pathologist,
        internal_medicine_specialist,
    ]

    return HierarchicalSwarm(
        name="Medical-Diagnosis-Hierarchical-Swarm",
        description="A hierarchical swarm for comprehensive medical diagnosis with specialized medical agents",
        director=chief_medical_officer,
        agents=medical_agents,
        max_loops=2,
        output_type="dict-all-except-first",
        reasoning_enabled=True,
    )


# Example 2: Legal Research Hierarchical Swarm
def create_legal_research_swarm():
    """
    Creates a hierarchical swarm for comprehensive legal research
    with specialized legal agents coordinated by a managing partner.
    """

    # Specialized legal agents
    corporate_lawyer = Agent(
        agent_name="Corporate-Law-Specialist",
        agent_description="Expert in corporate law, securities, and business transactions",
        system_prompt="""You are a senior corporate lawyer with expertise in:
        - Corporate governance and compliance
        - Securities law and regulations
        - Mergers and acquisitions
        - Contract law and commercial transactions
        - Business formation and structure
        
        Your responsibilities include:
        1. Analyzing corporate legal issues and compliance requirements
        2. Reviewing contracts and business agreements
        3. Advising on corporate governance matters
        4. Conducting due diligence for transactions
        5. Ensuring regulatory compliance
        
        You provide precise legal analysis with citations to relevant statutes and case law.""",
        model_name="claude-3-sonnet-20240229",
        max_loops=1,
        temperature=0.2,
    )

    litigation_attorney = Agent(
        agent_name="Litigation-Attorney",
        agent_description="Expert in civil litigation and dispute resolution",
        system_prompt="""You are a senior litigation attorney with expertise in:
        - Civil litigation and trial practice
        - Dispute resolution and mediation
        - Evidence analysis and case strategy
        - Legal research and brief writing
        - Settlement negotiations
        
        Your responsibilities include:
        1. Analyzing legal disputes and potential claims
        2. Developing litigation strategies and case theories
        3. Conducting legal research and precedent analysis
        4. Evaluating strengths and weaknesses of cases
        5. Recommending dispute resolution approaches
        
        You provide strategic legal analysis with case law support and risk assessment.""",
        model_name="claude-3-sonnet-20240229",
        max_loops=1,
        temperature=0.2,
    )

    regulatory_counsel = Agent(
        agent_name="Regulatory-Counsel",
        agent_description="Expert in regulatory compliance and government relations",
        system_prompt="""You are a senior regulatory counsel with expertise in:
        - Federal and state regulatory compliance
        - Administrative law and rulemaking
        - Government investigations and enforcement
        - Licensing and permitting requirements
        - Industry-specific regulations
        
        Your responsibilities include:
        1. Analyzing regulatory requirements and compliance obligations
        2. Monitoring regulatory developments and changes
        3. Advising on government relations strategies
        4. Conducting regulatory risk assessments
        5. Developing compliance programs and policies
        
        You provide comprehensive regulatory analysis with specific compliance recommendations.""",
        model_name="claude-3-sonnet-20240229",
        max_loops=1,
        temperature=0.2,
    )

    # Director agent
    managing_partner = Agent(
        agent_name="Managing-Partner",
        agent_description="Senior partner who coordinates comprehensive legal research and strategy",
        system_prompt="""You are a Managing Partner responsible for coordinating comprehensive 
        legal research and strategy. You oversee a team of legal specialists including:
        - Corporate Law Specialists
        - Litigation Attorneys
        - Regulatory Counsel
        
        Your role is to:
        1. Coordinate comprehensive legal analysis
        2. Assign specific legal research tasks to appropriate specialists
        3. Ensure all relevant legal domains are covered
        4. Synthesize findings from multiple legal experts
        5. Develop integrated legal strategies and recommendations
        6. Ensure adherence to professional standards and ethics
        
        You create specific, legally appropriate task assignments for each specialist.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        temperature=0.2,
    )

    legal_agents = [
        corporate_lawyer,
        litigation_attorney,
        regulatory_counsel,
    ]

    return HierarchicalSwarm(
        name="Legal-Research-Hierarchical-Swarm",
        description="A hierarchical swarm for comprehensive legal research with specialized legal agents",
        director=managing_partner,
        agents=legal_agents,
        max_loops=2,
        output_type="dict-all-except-first",
        reasoning_enabled=True,
    )


# Example 3: Software Development Hierarchical Swarm
def create_software_development_swarm():
    """
    Creates a hierarchical swarm for comprehensive software development
    with specialized development agents coordinated by a technical lead.
    """

    # Specialized development agents
    backend_developer = Agent(
        agent_name="Backend-Developer",
        agent_description="Expert in backend development, APIs, and system architecture",
        system_prompt="""You are a senior backend developer with expertise in:
        - Server-side programming and API development
        - Database design and optimization
        - System architecture and scalability
        - Cloud services and deployment
        - Security and performance optimization
        
        Your responsibilities include:
        1. Designing and implementing backend systems
        2. Creating RESTful APIs and microservices
        3. Optimizing database queries and performance
        4. Ensuring system security and reliability
        5. Implementing scalable architecture patterns
        
        You provide technical solutions with code examples and architectural recommendations.""",
        model_name="claude-3-sonnet-20240229",
        max_loops=1,
        temperature=0.4,
    )

    frontend_developer = Agent(
        agent_name="Frontend-Developer",
        agent_description="Expert in frontend development, UI/UX, and user interfaces",
        system_prompt="""You are a senior frontend developer with expertise in:
        - Modern JavaScript frameworks (React, Vue, Angular)
        - HTML5, CSS3, and responsive design
        - User experience and interface design
        - Performance optimization and accessibility
        - Testing and debugging frontend applications
        
        Your responsibilities include:
        1. Developing responsive user interfaces
        2. Implementing interactive frontend features
        3. Optimizing performance and user experience
        4. Ensuring cross-browser compatibility
        5. Following accessibility best practices
        
        You provide frontend solutions with code examples and UX considerations.""",
        model_name="claude-3-sonnet-20240229",
        max_loops=1,
        temperature=0.4,
    )

    devops_engineer = Agent(
        agent_name="DevOps-Engineer",
        agent_description="Expert in DevOps, CI/CD, and infrastructure automation",
        system_prompt="""You are a senior DevOps engineer with expertise in:
        - Continuous integration and deployment (CI/CD)
        - Infrastructure as Code (IaC) and automation
        - Containerization and orchestration (Docker, Kubernetes)
        - Cloud platforms and services (AWS, Azure, GCP)
        - Monitoring, logging, and observability
        
        Your responsibilities include:
        1. Designing and implementing CI/CD pipelines
        2. Automating infrastructure provisioning and management
        3. Ensuring system reliability and scalability
        4. Implementing monitoring and alerting systems
        5. Optimizing deployment and operational processes
        
        You provide DevOps solutions with infrastructure code and deployment strategies.""",
        model_name="claude-3-sonnet-20240229",
        max_loops=1,
        temperature=0.4,
    )

    # Director agent
    technical_lead = Agent(
        agent_name="Technical-Lead",
        agent_description="Senior technical lead who coordinates comprehensive software development",
        system_prompt="""You are a Technical Lead responsible for coordinating comprehensive 
        software development projects. You oversee a team of specialists including:
        - Backend Developers
        - Frontend Developers
        - DevOps Engineers
        
        Your role is to:
        1. Coordinate comprehensive software development efforts
        2. Assign specific development tasks to appropriate specialists
        3. Ensure all technical aspects are covered
        4. Synthesize technical requirements and solutions
        5. Develop integrated development strategies
        6. Ensure adherence to coding standards and best practices
        
        You create specific, technically appropriate task assignments for each specialist.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        temperature=0.4,
    )

    development_agents = [
        backend_developer,
        frontend_developer,
        devops_engineer,
    ]

    return HierarchicalSwarm(
        name="Software-Development-Hierarchical-Swarm",
        description="A hierarchical swarm for comprehensive software development with specialized development agents",
        director=technical_lead,
        agents=development_agents,
        max_loops=2,
        output_type="dict-all-except-first",
        reasoning_enabled=True,
    )


# Example usage and demonstration
if __name__ == "__main__":
    print("🏥 Medical Diagnosis Hierarchical Swarm Example")
    print("=" * 60)

    # Create medical diagnosis swarm
    medical_swarm = create_medical_diagnosis_swarm()

    medical_case = """
    Patient presents with:
    - 45-year-old male with chest pain and shortness of breath
    - Pain radiates to left arm and jaw
    - Elevated troponin levels
    - ECG shows ST-segment elevation
    - Family history of coronary artery disease
    - Current smoker, hypertension, diabetes
    
    Provide comprehensive diagnosis and treatment recommendations.
    """

    print("Running medical diagnosis analysis...")
    medical_result = medical_swarm.run(medical_case)
    print("Medical analysis complete!\n")

    print("⚖️ Legal Research Hierarchical Swarm Example")
    print("=" * 60)

    # Create legal research swarm
    legal_swarm = create_legal_research_swarm()

    legal_case = """
    A technology startup is planning to:
    - Raise Series A funding of $10M
    - Expand operations to European markets
    - Implement new data privacy policies
    - Negotiate strategic partnerships
    - Address potential IP disputes
    
    Provide comprehensive legal analysis and recommendations.
    """

    print("Running legal research analysis...")
    legal_result = legal_swarm.run(legal_case)
    print("Legal analysis complete!\n")

    print("💻 Software Development Hierarchical Swarm Example")
    print("=" * 60)

    # Create software development swarm
    dev_swarm = create_software_development_swarm()

    dev_project = """
    Develop a comprehensive e-commerce platform with:
    - User authentication and authorization
    - Product catalog and search functionality
    - Shopping cart and checkout process
    - Payment processing integration
    - Admin dashboard for inventory management
    - Mobile-responsive design
    - High availability and scalability requirements
    
    Provide technical architecture and implementation plan.
    """

    print("Running software development analysis...")
    dev_result = dev_swarm.run(dev_project)
    print("Software development analysis complete!\n")

    print("✅ All Hierarchical Swarm Examples Complete!")
    print("=" * 60)
