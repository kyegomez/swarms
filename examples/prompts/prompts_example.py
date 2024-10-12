from swarms.prompts.prompt import Prompt

# Example 1: Initializing a Financial Report Prompt
financial_prompt = Prompt(
    content="Q1 2024 Earnings Report: Initial Draft", autosave=True
)

# Output the initial state of the prompt
print("\n--- Example 1: Initializing Prompt ---")
print(f"Prompt ID: {financial_prompt.id}")
print(f"Content: {financial_prompt.content}")
print(f"Created At: {financial_prompt.created_at}")
print(f"Edit Count: {financial_prompt.edit_count}")
print(f"History: {financial_prompt.edit_history}")


# Example 2: Editing a Financial Report Prompt
financial_prompt.edit_prompt(
    "Q1 2024 Earnings Report: Updated Revenue Figures"
)

# Output the updated state of the prompt
print("\n--- Example 2: Editing Prompt ---")
print(f"Content after edit: {financial_prompt.content}")
print(f"Edit Count: {financial_prompt.edit_count}")
print(f"History: {financial_prompt.edit_history}")


# Example 3: Rolling Back to a Previous Version
financial_prompt.edit_prompt("Q1 2024 Earnings Report: Final Version")
financial_prompt.rollback(
    1
)  # Roll back to the second version (index 1)

# Output the state after rollback
print("\n--- Example 3: Rolling Back ---")
print(f"Content after rollback: {financial_prompt.content}")
print(f"Edit Count: {financial_prompt.edit_count}")
print(f"History: {financial_prompt.edit_history}")


# Example 4: Handling Invalid Rollback
print("\n--- Example 4: Invalid Rollback ---")
try:
    financial_prompt.rollback(
        5
    )  # Attempt an invalid rollback (out of bounds)
except IndexError as e:
    print(f"Error: {e}")


# Example 5: Preventing Duplicate Edits
print("\n--- Example 5: Preventing Duplicate Edits ---")
try:
    financial_prompt.edit_prompt(
        "Q1 2024 Earnings Report: Updated Revenue Figures"
    )  # Duplicate content
except ValueError as e:
    print(f"Error: {e}")


# Example 6: Retrieving the Prompt Content as a String
print("\n--- Example 6: Retrieving Prompt as String ---")
current_content = financial_prompt.get_prompt()
print(f"Current Prompt Content: {current_content}")


# Example 7: Simulating Financial Report Changes Over Time
print("\n--- Example 7: Simulating Changes Over Time ---")
# Initialize a new prompt representing an initial financial report draft
financial_prompt = Prompt(
    content="Q2 2024 Earnings Report: Initial Draft"
)

# Simulate several updates over time
financial_prompt.edit_prompt(
    "Q2 2024 Earnings Report: Updated Forecasts"
)
financial_prompt.edit_prompt(
    "Q2 2024 Earnings Report: Revenue Adjustments"
)
financial_prompt.edit_prompt("Q2 2024 Earnings Report: Final Review")

# Display full history
print(f"Final Content: {financial_prompt.content}")
print(f"Edit Count: {financial_prompt.edit_count}")
print(f"Edit History: {financial_prompt.edit_history}")
