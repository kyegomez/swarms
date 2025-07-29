import gradio as gr
import ai_gradio

finance_interface = gr.load(
    name="swarms:gpt-4-turbo",
    src=ai_gradio.registry,
    agent_name="Stock-Analysis-Agent",
    title="Finance Assistant",
    description="Expert financial analysis and advice tailored to your investment needs.",
)
finance_interface.launch()
