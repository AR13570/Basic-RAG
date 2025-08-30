from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.messages import SystemMessage

from agent.agent import CustomAgent
from agent.schema import AgentSchema
from knowledge_base.kb import KBase


system_message = SystemMessage(
    content="""You are a helpful assistant that helps students find information in their syllabus.

Guidelines:
- If context is provided (from a tool), always use it to answer the question.
- If no context is available, decide if you need to call a tool.
- If you don’t know based on the context, say "I don’t know based on the syllabus."
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
    description="Useful for when you need to find information in the syllabus to answer questions from students.",
)


agent_schema = AgentSchema(
    agent_name="syllabus_agent",
    model=ChatOllama(model="qwen3:1.7b"),
    system_prompt=system_message,  # now a SystemMessage, not ChatPromptTemplate
    tools=[retriever_tool],
)

agent = CustomAgent(agent_schema)


response = agent.invoke(
    {
        "question": "What is the syllabus for math?",
    }
)

print("\n[Final Response]")
print(response.content if hasattr(response, "content") else response)
for message in agent.messages:
    print(f"\n[{message.type}]", message.content[:50])