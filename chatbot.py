from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Customer service-oriented template
template = """
You are a helpful customer service assistant. Respond politely and provide accurate information or assistance as needed.

Here is the conversation history: {context}

Customer's question: {question}

Customer Service Response:
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model 

def handle_conversation():
    context = ""
    print("Welcome to the Customer Service ChatBot! How can I assist you today? Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Bot: Thank you for chatting with us. Have a great day!")
            break

        result = chain.invoke({"context": context, "question": user_input})
        print("Bot: ", result)
        context += f"\nCustomer: {user_input}\nCustomer Service: {result}"

if __name__ == "__main__":
    handle_conversation()
