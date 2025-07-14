# --- Streamlit App ---
import streamlit as st

from frontend import run_agent

# Set the title of the Streamlit app
st.title("🤖 Agent Chat")

# --- Sidebar for Thread ID Input ---
with st.sidebar:
    st.header("Configuration")
    # Create a text input for the user to enter a thread_id
    thread_id = st.text_input("Enter Thread ID", value="default-thread-123")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---

# Initialize the chat history in Streamlit's session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages from the history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from the chat input box at the bottom of the screen
if prompt := st.chat_input("What is your question?"):

    # Check if a thread_id has been provided
    if not thread_id:
        st.warning("Please enter a Thread ID in the sidebar before starting the chat.")
        st.stop()

    # Add the user's message to the chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the agent's response
    with st.chat_message("assistant"):
        # Use a spinner to indicate that the agent is thinking
        with st.spinner("Thinking..."):
            # Call your agent function with the user's question and the thread_id
            response = run_agent(question=prompt, thread_id=thread_id)

            # Display the agent's response
            st.markdown(response)

    # Add the agent's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})