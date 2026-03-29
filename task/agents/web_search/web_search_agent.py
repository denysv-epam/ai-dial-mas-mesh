from task.agents.base_agent import BaseAgent
from task.agents.web_search._prompts import SYSTEM_PROMPT
from task.tools.base_tool import BaseTool

class WebSearchAgent(BaseAgent):
    """Web-search-focused agent that wraps BaseAgent with a fixed prompt.

    The agent delegates tool execution to the BaseAgent while applying
    the web search system prompt and a provided set of tools.
    """

    def __init__(self, endpoint: str, tools: list[BaseTool]):
        """Initialize the web search agent with endpoint and tools.

        Args:
            endpoint: Base URL for DIAL core or adapter.
            tools: Tool instances available to the agent.
        """
        super().__init__(endpoint=endpoint, system_prompt=SYSTEM_PROMPT, tools=tools)
