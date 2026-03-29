import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agents.calculations.calculations_agent import CalculationsAgent
from task.agents.calculations.tools.simple_calculator_tool import SimpleCalculatorTool
from task.tools.base_tool import BaseTool
from task.agents.calculations.tools.py_interpreter.python_code_interpreter_tool import PythonCodeInterpreterTool
from task.tools.deployment.content_management_agent_tool import ContentManagementAgentTool
from task.tools.deployment.web_search_agent_tool import WebSearchAgentTool
from task.utils.constants import DIAL_ENDPOINT, DEPLOYMENT_NAME

_PYINTERPRETER_MCP_URL = os.getenv("PYINTERPRETER_MCP_URL", "http://localhost:8050/mcp")


class CalculationsApplication(ChatCompletion):
    """ChatCompletion app wiring for the Calculations Agent."""

    def __init__(self):
        """Initialize the application with lazily created tools.

        Tools are created on the first request so that async MCP setup
        happens inside the event loop rather than at import time.
        """
        self.tools: list[BaseTool] = []

    async def _create_tools(self) -> list[BaseTool]:
        """Construct the tool set used by the Calculations Agent.

        The tool list combines local utilities (calculator and Python
        interpreter) with mesh tools that call other agents via DIAL.
        """
        tools: list[BaseTool] = [
            SimpleCalculatorTool(),
            await PythonCodeInterpreterTool.create(
                mcp_url=_PYINTERPRETER_MCP_URL,
                tool_name="execute_code",
                dial_endpoint=DIAL_ENDPOINT,
            ),
            ContentManagementAgentTool(endpoint=DIAL_ENDPOINT),
            WebSearchAgentTool(endpoint=DIAL_ENDPOINT),
        ]
        return tools

    async def chat_completion(self, request: Request, response: Response) -> None:
        """Handle a chat completion by delegating to CalculationsAgent.

        This method ensures the tool list is ready, creates a single
        streaming choice, and then forwards the request to the agent.
        """
        if not self.tools:
            self.tools = await self._create_tools()

        with response.create_single_choice() as choice:
            agent = CalculationsAgent(endpoint=DIAL_ENDPOINT, tools=self.tools)
            await agent.handle_request(
                choice=choice,
                deployment_name=DEPLOYMENT_NAME,
                request=request,
                response=response,
            )


app: DIALApp = DIALApp()
agent_app = CalculationsApplication()
app.add_chat_completion("calculations-agent", agent_app)


if __name__ == "__main__":
    uvicorn.run(app, port=5001, host="0.0.0.0")
