from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.messages import SystemMessage
from agent.agent import CustomAgent
from agent.schema import AgentSchema
from knowledge_base.kb import KBase
from utils.streamer import streamer
from utils.non_streamer import non_streamer
import logging

logging.disable()

system_message = SystemMessage(
    content="""You are a helpful assistant that helps students find information in their syllabus. If you dont know the answer check if there is any existing tool that can help you.
            Guidelines:
            - If no context is available, decide if you need to call a tool.
            - If the user asks for multiple things, break it down into its individual parts and answer each part separately. Each part can require a tool call, so make sure to check for that.
            - If context is provided (from a tool), always use it to answer the question.
            - If you don't know based on the context, and are also not able to use any tool to find the answer, only then say "I don't know based on the syllabus."
            """
)


kb = KBase(
    "./source_data/syllabus",
    "./vector_db",
    "syllabus_docs",
    OllamaEmbeddings(model="mxbai-embed-large"),
)

# vector_store = kb.embed_and_store()
vector_store = kb.get_vector_store()
retriever = kb.get_retriever(vector_store, k=4)

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="syllabus_retriever",
    description="Useful for when you need to find information in the syllabus to answer questions from students. "
    "Always use this tool if you need to find any information related to the syllabus. "
    "The input to this tool would only be related to a single subject at a time. If multiple subjects are mentioned, "
    "break down the question and use the tool multiple times, once for each subject.",
)


agent_schema = AgentSchema(
    agent_name="syllabus_agent",
    model=ChatOllama(model="qwen3:1.7b"),
    system_prompt=system_message,  # now a SystemMessage, not ChatPromptTemplate
    tools=[retriever_tool],
)

agent = CustomAgent(agent_schema)


while True:
    mode = input(
        "\nChoose mode: (1) Non-Streaming, (2) Streaming (or 'exit' to quit): "
    )
    if mode.lower() == "exit":
        exit()
    question = input("\nEnter your question (or 'exit' to quit): ")
    if question.lower() == "exit":
        exit()
    if mode == "1":
        non_streamer(agent=agent, question=question)
    elif mode == "2":
        streamer(agent=agent, question=question)

    agent.clear_messages()
