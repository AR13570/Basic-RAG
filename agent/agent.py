from .schema import AgentSchema
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage


class CustomAgent:
    def __init__(
        self, agent_schema: AgentSchema, verbose: bool = True, max_depth: int = 2
    ):
        self.system_prompt = agent_schema.system_prompt
        self.model = agent_schema.model.bind_tools(agent_schema.tools)
        self.tools = {tool.name: tool for tool in agent_schema.tools}
        self.verbose = verbose
        self.messages = []
        self.max_depth = max_depth

    def _call_agent(self):
        """Call model with full conversation history (system + messages)."""
        return self.model.invoke([self.system_prompt] + self.messages)

    def invoke(self, inputs: dict, depth: int = 0):
        if depth > self.max_depth:
            return AIMessage(content="Max recursion depth reached. Stopping.")

        if depth==0:
            # Add user message
            self.messages.append(HumanMessage(content=inputs["question"]))

        # Call model
        result = self._call_agent()
        print("\n[Agent Response]",result)
        self.messages.append(result)

        if hasattr(result, "tool_calls") and result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["args"]


                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    tool_result = tool.invoke(tool_input)

                    # Flatten retriever docs
                    if isinstance(tool_result, list):
                        tool_result = "\n\n".join(
                            [doc.page_content for doc in tool_result]
                        )

                    self.messages.append(
                        ToolMessage(
                            content=str(tool_result), tool_call_id=tool_call["id"]
                        )
                    )

                    # 🔄 Recurse with updated messages
                    return self.invoke(inputs, depth + 1)

                else:
                    if self.verbose:
                        print(f"[Warning] Tool {tool_name} not found.")

        return result
