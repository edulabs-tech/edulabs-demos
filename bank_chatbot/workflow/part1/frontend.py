from bank_chatbot.workflow.part1.backend import stream_graph_updates

if __name__ == '__main__':
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
