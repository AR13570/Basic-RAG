from agent.agent import CustomAgent


def non_streamer(agent: CustomAgent, question: str):
    response = agent.invoke({"question": question})
    print("\n[Response]")
    print(response.content if hasattr(response, "content") else response)
    print("\n[End of Response]")
