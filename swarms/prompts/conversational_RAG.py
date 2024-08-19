from langchain.prompts.prompt import PromptTemplate

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

QA_CONDENSE_TEMPLATE_STR = (
    "Given the following Chat History and a Follow Up Question, "
    "rephrase the follow up question to be a new Standalone Question, "
    "but make sure the new question is still asking for the same "
    "information as the original follow up question. Respond only "
    " with the new Standalone Question. \n"
    "Chat History: \n"
    "{chat_history} \n"
    "Follow Up Question: {input} \n"
    "Standalone Question:"
)

CONDENSE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    f"{B_INST}{B_SYS}{QA_CONDENSE_TEMPLATE_STR.strip()}{E_SYS}{E_INST}"
)

QA_PROMPT_TEMPLATE_STR = (
    "HUMAN: \n You are a helpful AI assistant.  "
    "Use the following context and chat history to answer the "
    "question at the end with a helpful answer.  "
    "Get straight to the point and always think things through step-by-step before answering.  "
    "If you don't know the answer, just say 'I don't know'; "
    "don't try to make up an answer. \n\n"
    "{context}\n"
    "{chat_history}\n"
    "{input}\n\n"
    "AI:  Here is the most relevant sentence in the context:  \n"
)

QA_PROMPT_TEMPLATE = PromptTemplate.from_template(
    f"{B_INST}{B_SYS}{QA_PROMPT_TEMPLATE_STR.strip()}{E_SYS}{E_INST}"
)

DOCUMENT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)

_STUFF_PROMPT_TEMPLATE_STR = "Summarize the following context: {context}"

STUFF_PROMPT_TEMPLATE = PromptTemplate.from_template(
    f"{B_INST}{B_SYS}{_STUFF_PROMPT_TEMPLATE_STR.strip()}{E_SYS}{E_INST}"
)

_SUMMARIZER_SYS_TEMPLATE = (
    B_INST
    + B_SYS
    + """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.
    EXAMPLE
    Current summary:
    The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.
    New lines of conversation:
    Human: Why do you think artificial intelligence is a force for good?
    AI: Because artificial intelligence will help humans reach their full potential.
    New summary:
    The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
    END OF EXAMPLE"""
    + E_SYS
    + E_INST
)

_SUMMARIZER_INST_TEMPLATE = (
    B_INST
    + """Current summary:
    {summary}
    New lines of conversation:
    {new_lines}
    New summary:"""
    + E_INST
)

SUMMARY_PROMPT_TEMPLATE = PromptTemplate.from_template(
    template=(_SUMMARIZER_SYS_TEMPLATE + "\n" + _SUMMARIZER_INST_TEMPLATE).strip()
)