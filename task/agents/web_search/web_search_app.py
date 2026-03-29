import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agents.web_search.web_search_agent import WebSearchAgent
from task.tools.base_tool import BaseTool
from task.tools.deployment.calculations_agent_tool import CalculationsAgentTool
from task.tools.deployment.content_management_agent_tool import ContentManagementAgentTool
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool import MCPTool
from task.utils.constants import DIAL_ENDPOINT, DEPLOYMENT_NAME

_DDG_MCP_URL = os.getenv('DDG_MCP_URL', "http://localhost:8051/mcp")


class WebSearchApplication(ChatCompletion):
    """ChatCompletion app wiring for the Web Search Agent."""

    def __init__(self):
        """Initialize the application with lazily created tools.

        Tools are created on the first request so that async MCP setup
        occurs within the running event loop.
        """
        self.tools: list[BaseTool] = []

    async def _get_mcp_tools(self, url: str) -> list[BaseTool]:
        """Fetch MCP tools from the given MCP server URL.

        Args:
            url: MCP server endpoint.

        Returns:
            A list of BaseTool instances wrapping MCP tools.
        """
        tools: list[BaseTool] = []
        mcp_client = await MCPClient.create(url)
        for tool in await mcp_client.get_tools():
            tools.append(MCPTool(mcp_client, tool))
        return tools

    async def _create_tools(self) -> list[BaseTool]:
        """Construct the tool set used by the Web Search Agent.

        This combines MCP search tools with mesh tools that route
        to other agents for calculations or content handling.
        """
        tools: list[BaseTool] = []
        tools.extend(await self._get_mcp_tools(_DDG_MCP_URL))
        tools.append(CalculationsAgentTool(endpoint=DIAL_ENDPOINT))
        tools.append(ContentManagementAgentTool(endpoint=DIAL_ENDPOINT))
        return tools

    async def chat_completion(self, request: Request, response: Response) -> None:
        """Handle a chat completion by delegating to WebSearchAgent.

        Ensures the tool list is ready, creates a single streaming
        choice, and forwards the request to the agent.
        """
        if not self.tools:
            self.tools = await self._create_tools()

        with response.create_single_choice() as choice:
            agent = WebSearchAgent(endpoint=DIAL_ENDPOINT, tools=self.tools)
            await agent.handle_request(
                choice=choice,
                deployment_name=DEPLOYMENT_NAME,
                request=request,
                response=response,
            )


app: DIALApp = DIALApp()
agent_app = WebSearchApplication()
app.add_chat_completion("web-search-agent", agent_app)


if __name__ == "__main__":
    uvicorn.run(app, port=5003, host="0.0.0.0")
