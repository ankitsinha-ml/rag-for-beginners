from retrieval_pipeline import get_context
from answer_generation import generate_answer

history = ""

print("RAG Chatbot ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ").strip()

    if query.lower() == "exit":
        print("Goodbye!")
        break

    if not query:
        continue

    context = get_context(query)

    if not context:
        print("Bot: I couldn't find any relevant information.\n")
        continue

    response = generate_answer(query, context, history)

    print(f"Bot: {response}\n")

    history += f"Human: {query}\nBot: {response}\n"
    history_lines = history.strip().split("\n")

    if len(history_lines) > 6:
        history = "\n".join(history_lines[-6:]) + "\n"