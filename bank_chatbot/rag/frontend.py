import gradio as gr

from bank_chatbot.rag.backend import stream_rag_chain

if __name__ == "__main__":
    with gr.Blocks(fill_height=True) as demo:
        gr.ChatInterface(
            stream_rag_chain,
            type="messages",
            multimodal=False,
            theme="soft"
        )
    demo.launch(share=False)


# how much mortgage i can get for 3000000 ils apartment?