import gradio as gr

from bank_chatbot.basics.general_info import invoke_llm, stream_llm, invoke_with_trim

if __name__ == "__main__":
    with gr.Blocks(fill_height=True) as demo:
        model = gr.Dropdown(["Gemini", "Open AI"], label="Select model:")
        language = gr.Dropdown(["English", "Hebrew"], label="Select language:")

        gr.ChatInterface(
            invoke_llm,
            # stream_llm,
            # invoke_with_trim,
            type="messages",
            multimodal=False,
            theme="soft",
            additional_inputs=[
                language,
                model
            ],

        )
    demo.launch(share=False)