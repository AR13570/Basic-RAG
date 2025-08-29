from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = ChatOllama(model="llama3.2")

template = """
            You are a helpful answering bot that reads the relevant reviews and answers the question given by the user.
            Here are the relevant reviews: {reviews}
            Here is the question: {question}
            if there are no relevant reviews tell the user this and dont give an answer
            """
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input("Enter the question(enter q to quit): ")
    if question == "q":
        break
    reviews = retriever.invoke(question)
    print(reviews)
    results = chain.invoke({"reviews": reviews, "question": question})
    print("\n\n", results.content)
