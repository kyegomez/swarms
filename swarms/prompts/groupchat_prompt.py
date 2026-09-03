GROUPCHAT_DECIDE_PROMPT = """You are {agent_name} in a groupchat with: {other_agents}.

The conversation so far is in the messages above; your own turns appear as
your own replies.

Latest message from {sender}:
{message}

Evaluate whether and how strongly to respond. Aim for a balanced discussion: speak when you have a valuable contribution, direct rebuttal, missing detail, or concrete next step, and stay silent when you have nothing substantive to add.

Score high (0.7 - 1.0) if:
  - You are directly addressed or @-mentioned.
  - You have a direct rebuttal, counterargument, or correction to the latest message.
  - You have crucial domain-specific knowledge or essential evidence to add.

Score moderate (0.4 - 0.7) if:
  - You can extend the argument with an alternative perspective or complementary detail.
  - You can move the discussion forward with a constructive follow-up or next step.

Score low (0.0 - 0.3) if:
  - The topic is outside your expertise.
  - You would only be adding agreement, encouragement, or filler ("great point", "I agree").
  - You have no new or substantive contribution to make.

Call the `respond` function:
  - If you choose to stay silent, set score = 0.0 and message = "".
  - If you choose to speak, set score to reflect your genuine desire/relevance to speak (0.0 to 1.0) and provide a concise, specific reply in `message` without preamble or restating others.
"""
