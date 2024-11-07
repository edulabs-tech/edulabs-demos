import gradio as gr
from gradio import ChatMessage
from langchain_core.messages import HumanMessage, AIMessage

from bank_chatbot.tools.backend import agent_executor


def interact_with_langchain_agent(prompt, messages):
    messages.append(ChatMessage(role="user", content=prompt))
    for event in agent_executor.stream(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": "abc123"}},
        stream_mode="values"
    ):
        last_message = event["messages"][-1]
        if isinstance(last_message, AIMessage):
            for tool_call in last_message.tool_calls:
                messages.append(ChatMessage(
                    role="assistant",
                    content="",
                    metadata={"title": f"🛠️ Using tool {tool_call['name']}"}
                ))
                yield gr.update(value=""), messages

            if len(last_message.content) > 0:
                messages.append(ChatMessage(role="assistant", content=last_message.content))
                yield gr.update(value=""), messages


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("# Chat with a LangChain Agent 🦜⛓️ and see its thoughts 💭")
        chatbot = gr.Chatbot(
            type="messages",
            label="Agent",
            avatar_images=(
                None,
                "https://em-content.zobj.net/source/twitter/141/parrot_1f99c.png",
            ),
        )
        textbox = gr.Textbox(lines=1, label="Chat Message")
        textbox.submit(interact_with_langchain_agent, [textbox, chatbot], [textbox, chatbot])

    demo.launch()
