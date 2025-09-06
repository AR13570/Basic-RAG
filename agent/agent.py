from .schema import AgentSchema
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage


class CustomAgent:
    def __init__(
        self, agent_schema: AgentSchema, verbose: bool = True, max_depth: int = 4
    ):
        self.system_prompt = agent_schema.system_prompt
        self.model = agent_schema.model.bind_tools(agent_schema.tools)
        self.tools = {tool.name: tool for tool in agent_schema.tools}
        self.verbose = verbose
        self.messages = []
        self.max_depth = max_depth

    def clear_messages(self):
        self.messages = []

    def stream(self, inputs: dict, depth: int = 0):
        if depth > self.max_depth:
            yield AIMessage(content="Max recursion depth reached. Stopping.")
            return
        if depth == 0:
            # Add user message
            self.messages.append(HumanMessage(content=inputs["question"]))
        print("\n[At depth]", depth)
        generator = self.model.stream([self.system_prompt] + self.messages)
        tool_calls = []
        result_text = ""
        try:
            for result in generator:
                yield result
                result_text += result.content
                if hasattr(result, "tool_calls") and result.tool_calls:
                    print("\n[Tool Calls Detected]", result.tool_calls)
                    tool_calls.extend(result.tool_calls)
        except StopIteration:
            pass

        self.messages.append(AIMessage(content=result_text, tool_calls=tool_calls))

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]

            if tool_name in self.tools:
                tool = self.tools[tool_name]
                print("\n[Invoking Tool]", tool_name, tool_input)
                tool_result = tool.invoke(tool_input)
                tool_message = ToolMessage(
                    content=str(tool_result), tool_call_id=tool_call["id"]
                )
                print("\n[Tool Result]", tool_result[:100], "...(truncated)")
                self.messages.append(tool_message)
            else:
                print(f"\n[Warning] Tool {tool_name} not found.")
        if len(tool_calls) == 0:
            print("\n[No Tool Calls, Returning Result]")
            return
        # call the model again with the tool results
        yield from self.stream(inputs, depth + 1)
        return

    def invoke(self, inputs: dict, depth: int = 0):
        if depth > self.max_depth:
            return AIMessage(content="Max recursion depth reached. Stopping.")

        if depth == 0:
            # Add user message
            self.messages.append(HumanMessage(content=inputs["question"]))

        print("\n[At depth]", depth)
        # Call model
        result = self.model.invoke([self.system_prompt] + self.messages)
        print("\n[Agent Response]", result)
        self.messages.append(result)

        if hasattr(result, "tool_calls") and result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["args"]
                print("\n[Invoking Tool]", tool_name, tool_input)
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    tool_result = tool.invoke(tool_input)
                    print("\n[Tool Result]", tool_result[:100], "...(truncated)")

                    self.messages.append(
                        ToolMessage(
                            content=str(tool_result), tool_call_id=tool_call["id"]
                        )
                    )

                else:
                    print(f"\n[Warning] Tool {tool_name} not found.")
            else:
                # call the model again with the tool results
                return self.invoke(inputs, depth + 1)
        # no tool calls, return the result
        print("\n[No Tool Calls, Returning Result]")
        return result
