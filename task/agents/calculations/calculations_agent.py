from task.agents.base_agent import BaseAgent
from task.agents.calculations._prompts import SYSTEM_PROMPT
from task.tools.base_tool import BaseTool


class CalculationsAgent(BaseAgent):
    """Calculations-focused agent that wraps BaseAgent with a fixed prompt.

    The agent delegates tool execution to the BaseAgent while applying
    the calculations system prompt and a provided set of tools.
    """

    def __init__(self, endpoint: str, tools: list[BaseTool]):
        """Initialize the calculations agent with endpoint and tools.

        Args:
            endpoint: Base URL for DIAL core or adapter.
            tools: Tool instances available to the agent.
        """
        super().__init__(endpoint=endpoint, system_prompt=SYSTEM_PROMPT, tools=tools)
