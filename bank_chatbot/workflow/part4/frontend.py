from bank_chatbot.workflow.part4.backend import stream_graph_updates

if __name__ == '__main__':
    while True:
        # thread_id = input("Session ID: ")
        thread_id = "1"
        user_input = input("User message: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input, thread_id)
