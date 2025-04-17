
from swarms.structs.agent import Agent
from typing import Dict, List

class HealthcareDiagnosticSystem:
    def __init__(self):
        self.primary_diagnostician = Agent(
            agent_name="Primary-Diagnostician",
            agent_description="Primary diagnostic analysis specialist",
            system_prompt="""You are a primary diagnostician expert in:
            1. Initial Symptom Analysis
            2. Patient History Evaluation
            3. Preliminary Diagnosis Formation
            4. Risk Factor Assessment
            5. Treatment Priority Determination""",
            max_loops=3,
            model_name="gpt-4"
        )
        
        self.specialist_consultant = Agent(
            agent_name="Specialist-Consultant",
            agent_description="Specialized medical consultation expert",
            system_prompt="""You are a medical specialist focusing on:
            1. Complex Case Analysis
            2. Specialized Treatment Planning
            3. Comorbidity Assessment
            4. Treatment Risk Evaluation
            5. Advanced Diagnostic Interpretation""",
            max_loops=3,
            model_name="gpt-4"
        )
        
        self.treatment_coordinator = Agent(
            agent_name="Treatment-Coordinator",
            agent_description="Treatment planning and coordination specialist",
            system_prompt="""You are a treatment coordination expert specializing in:
            1. Treatment Plan Development
            2. Care Coordination
            3. Resource Allocation
            4. Recovery Timeline Planning
            5. Follow-up Protocol Design""",
            max_loops=3,
            model_name="gpt-4"
        )

    def process_case(self, patient_data: Dict) -> Dict:
        # Initial diagnosis
        primary_assessment = self.primary_diagnostician.run(
            f"Perform initial diagnosis: {patient_data}"
        )
        
        # Specialist consultation
        specialist_review = self.specialist_consultant.run(
            f"Review case with initial assessment: {primary_assessment}"
        )
        
        # Treatment planning
        treatment_plan = self.treatment_coordinator.run(
            f"Develop treatment plan based on: Primary: {primary_assessment}, Specialist: {specialist_review}"
        )
        
        return {
            "initial_assessment": primary_assessment,
            "specialist_review": specialist_review,
            "treatment_plan": treatment_plan
        }

# Usage
diagnostic_system = HealthcareDiagnosticSystem()
