from agent.agent import CustomAgent


def streamer(agent: CustomAgent, question: str):
    print("\n[Streaming Response]")
    for chunk in agent.stream({"question": question}):
        print(chunk.content, end="", flush=True)
    print("\n[End of Response]")
