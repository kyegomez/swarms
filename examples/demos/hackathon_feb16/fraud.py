import csv
import json
from swarms import Agent


###############################################################################
# FraudClassifier Class Definition
###############################################################################
class FraudClassifier:
    def __init__(self, model_name="gpt-4o-mini"):
        """
        Initialize the system prompts and all agent instances.
        """

        # ------------------ Boss Agent Prompt ------------------ #
        self.BOSS_AGENT_SYS_PROMPT = """
        You are the Boss Agent. Your role is to orchestrate fraud analysis.
        
        First, you will receive a full CSV row as a string. Your task is to parse the row
        into its component fields: declared_country, ip_country, phone_carrier_country, ip_address, 
        known_blacklisted_regions, user_name, email, account_name, payment_info_name, account_history_notes.
        
        Then, you should instruct which specialized agents to call:
          - For location data (declared_country, ip_country, phone_carrier_country): Geolocation Agent.
          - For IP data (ip_address, known_blacklisted_regions): IP Agent.
          - For account data (user_name, email, account_name, payment_info_name, account_history_notes): Email Agent.
        
        Respond with your instructions in JSON format like:
        {
          "geolocation_data": "Concise location info",
          "ip_data": "Concise ip info",
          "email_data": "Concise account info"
        }
        
        After you receive the specialized agents’ responses, you will be given the full row
        again along with the sub-agent results. Then produce a final JSON in the following format:
        {
          "final_suspicious": bool,
          "details": [
            { "agent_name": "GeolocationAgent", "is_suspicious": bool, "reason": "..." },
            { "agent_name": "IPAgent", "is_suspicious": bool, "reason": "..." },
            { "agent_name": "EmailAgent", "is_suspicious": bool, "reason": "..." }
          ],
          "overall_reason": "Short summary"
        }
        """

        # ------------------ Specialized Agent Prompts ------------------ #
        self.GEOLOCATION_AGENT_SYS_PROMPT = """
        You are the Geolocation Agent.
        Your input is location-related data (declared_country, ip_country, phone_carrier_country).
        Decide if there is a suspicious mismatch.
        Return a JSON in the following format:
        {
          "is_suspicious": true or false,
          "reason": "Short reason"
        }
        """

        self.IP_AGENT_SYS_PROMPT = """
        You are the IP Agent.
        Your input includes ip_address and known_blacklisted_regions.
        Decide if the IP is suspicious (e.g., blacklisted or unusual).
        Return a JSON in the following format:
        {
          "is_suspicious": true or false,
          "reason": "Short reason"
        }
        """

        self.EMAIL_AGENT_SYS_PROMPT = """
        You are the Email/Account Agent.
        Your input includes user_name, email, account_name, payment_info_name, and account_history_notes.
        Decide if the account data is suspicious (e.g., name mismatches or new account flags).
        Return a JSON in the following format:
        {
          "is_suspicious": true or false,
          "reason": "Short reason"
        }
        """

        # ------------------ Initialize Agents ------------------ #
        self.boss_agent = Agent(
            agent_name="Boss-Agent",
            system_prompt=self.BOSS_AGENT_SYS_PROMPT,
            model_name=model_name,
            max_loops=2,
            autosave=False,
            dashboard=False,
            verbose=False,
            dynamic_temperature_enabled=False,
            saved_state_path="boss_agent.json",
            user_name="swarms_corp",
            retry_attempts=1,
            context_length=200000,
            return_step_meta=False,
            output_type="json",  # Expect JSON output
            streaming_on=False,
        )

        self.geolocation_agent = Agent(
            agent_name="Geolocation-Agent",
            system_prompt=self.GEOLOCATION_AGENT_SYS_PROMPT,
            model_name=model_name,
            max_loops=1,
            autosave=False,
            dashboard=False,
            verbose=False,
            dynamic_temperature_enabled=False,
            saved_state_path="geolocation_agent.json",
            user_name="swarms_corp",
            retry_attempts=1,
            context_length=200000,
            return_step_meta=False,
            output_type="json",
            streaming_on=False,
        )

        self.ip_agent = Agent(
            agent_name="IP-Agent",
            system_prompt=self.IP_AGENT_SYS_PROMPT,
            model_name=model_name,
            max_loops=1,
            autosave=False,
            dashboard=False,
            verbose=False,
            dynamic_temperature_enabled=False,
            saved_state_path="ip_agent.json",
            user_name="swarms_corp",
            retry_attempts=1,
            context_length=200000,
            return_step_meta=False,
            output_type="json",
            streaming_on=False,
        )

        self.email_agent = Agent(
            agent_name="Email-Agent",
            system_prompt=self.EMAIL_AGENT_SYS_PROMPT,
            model_name=model_name,
            max_loops=1,
            autosave=False,
            dashboard=False,
            verbose=False,
            dynamic_temperature_enabled=False,
            saved_state_path="email_agent.json",
            user_name="swarms_corp",
            retry_attempts=1,
            context_length=200000,
            return_step_meta=False,
            output_type="json",
            streaming_on=False,
        )

    def classify_row(self, row: dict) -> dict:
        """
        For a given CSV row (as a dict):
          1. Concatenate the entire row into a single string.
          2. Send that string to the Boss Agent to instruct how to parse and dispatch to sub‑agents.
          3. Call the specialized agents with the appropriate data.
          4. Send the sub‑agent results back to the Boss Agent for a final decision.
        """
        # (a) Concatenate entire row into one string.
        row_string = " ".join(f"{k}: {v}" for k, v in row.items())

        # (b) Send row to Boss Agent for parsing/instructions.
        initial_prompt = f"""
        Here is a CSV row data:
        {row_string}
        
        Please parse the row into its fields:
        declared_country, ip_country, phone_carrier_country, ip_address, known_blacklisted_regions, user_name, email, account_name, payment_info_name, account_history_notes.
        Then, provide your instructions (in JSON) on what to send to the sub-agents.
        For example:
        {{
          "geolocation_data": "declared_country, ip_country, phone_carrier_country",
          "ip_data": "ip_address, known_blacklisted_regions",
          "email_data": "user_name, email, account_name, payment_info_name, account_history_notes"
        }}
        """
        boss_instructions_raw = self.boss_agent.run(initial_prompt)
        try:
            boss_instructions = json.loads(boss_instructions_raw)
        except Exception:
            # If parsing fails, we fall back to manually constructing the sub-agent inputs.
            boss_instructions = {
                "geolocation_data": f"declared_country: {row.get('declared_country','')}, ip_country: {row.get('ip_country','')}, phone_carrier_country: {row.get('phone_carrier_country','')}",
                "ip_data": f"ip_address: {row.get('ip_address','')}, known_blacklisted_regions: {row.get('known_blacklisted_regions','')}",
                "email_data": f"user_name: {row.get('user_name','')}, email: {row.get('email','')}, account_name: {row.get('account_name','')}, payment_info_name: {row.get('payment_info_name','')}, account_history_notes: {row.get('account_history_notes','')}",
            }

        # (c) Call specialized agents using either the Boss Agent's instructions or defaults.
        geo_result = self.geolocation_agent.run(
            boss_instructions.get("geolocation_data", "")
        )
        ip_result = self.ip_agent.run(
            boss_instructions.get("ip_data", "")
        )
        email_result = self.email_agent.run(
            boss_instructions.get("email_data", "")
        )

        # (d) Consolidate specialized agent results as JSON.
        specialized_results = {
            "GeolocationAgent": geo_result,
            "IPAgent": ip_result,
            "EmailAgent": email_result,
        }
        specialized_results_json = json.dumps(specialized_results)

        # (e) Send the original row data and the specialized results back to the Boss Agent.
        final_prompt = f"""
        Here is the original CSV row data:
        {row_string}

        Here are the results from the specialized agents:
        {specialized_results_json}

        Based on this information, produce the final fraud classification JSON in the format:
        {{
          "final_suspicious": bool,
          "details": [
            {{ "agent_name": "GeolocationAgent", "is_suspicious": bool, "reason": "..." }},
            {{ "agent_name": "IPAgent", "is_suspicious": bool, "reason": "..." }},
            {{ "agent_name": "EmailAgent", "is_suspicious": bool, "reason": "..." }}
          ],
          "overall_reason": "Short summary"
        }}
        """
        final_result_raw = self.boss_agent.run(final_prompt)
        try:
            final_result = json.loads(final_result_raw)
        except Exception:
            final_result = {"final_result_raw": final_result_raw}
        return final_result

    def classify_csv(self, csv_file_path: str):
        """
        Load a CSV file, iterate over each row, run the classification, and return the results.
        """
        results = []
        with open(csv_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                result = self.classify_row(row)
                results.append(result)
        return results


###############################################################################
# Example Usage
###############################################################################
if __name__ == "__main__":
    # Create an instance of the FraudClassifier.
    classifier = FraudClassifier()

    # Specify your CSV file (e.g., "fraud_data.csv")
    csv_file = "fraud_data.csv"
    all_reports = classifier.classify_csv(csv_file)

    # Print the final fraud classification for each row.
    for idx, report in enumerate(all_reports, start=1):
        print(
            f"Row {idx} classification:\n{json.dumps(report, indent=2)}\n"
        )
