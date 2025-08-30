from pydantic import BaseModel
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage


class AgentSchema(BaseModel):
    agent_name: str
    model: BaseChatModel
    system_prompt: BaseMessage
    tools: list = []
