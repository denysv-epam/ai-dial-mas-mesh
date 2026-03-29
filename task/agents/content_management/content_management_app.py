import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agents.content_management.content_management_agent import ContentManagementAgent
from task.agents.content_management.tools.files.file_content_extraction_tool import FileContentExtractionTool
from task.agents.content_management.tools.rag.document_cache import DocumentCache
from task.agents.content_management.tools.rag.rag_tool import RagTool
from task.tools.base_tool import BaseTool
from task.tools.deployment.calculations_agent_tool import CalculationsAgentTool
from task.tools.deployment.web_search_agent_tool import WebSearchAgentTool
from task.utils.constants import DIAL_ENDPOINT, DEPLOYMENT_NAME


class ContentManagementApplication(ChatCompletion):
    """ChatCompletion app wiring for the Content Management Agent."""

    def __init__(self):
        """Initialize the application with lazily created tools.

        Tools are created on the first request to keep async operations
        inside the event loop.
        """
        self.tools: list[BaseTool] = []

    async def _create_tools(self) -> list[BaseTool]:
        """Construct the tool set used by the Content Management Agent.

        The tool list includes file extraction and RAG tooling plus
        mesh tools that route to other agents.
        """
        document_cache = DocumentCache.create()
        tools: list[BaseTool] = [
            FileContentExtractionTool(endpoint=DIAL_ENDPOINT),
            RagTool(
                endpoint=DIAL_ENDPOINT,
                deployment_name=DEPLOYMENT_NAME,
                document_cache=document_cache,
            ),
            CalculationsAgentTool(endpoint=DIAL_ENDPOINT),
            WebSearchAgentTool(endpoint=DIAL_ENDPOINT),
        ]
        return tools

    async def chat_completion(self, request: Request, response: Response) -> None:
        """Handle a chat completion by delegating to ContentManagementAgent.

        This method ensures the tool list is ready, creates a single
        streaming choice, and then forwards the request to the agent.
        """
        if not self.tools:
            self.tools = await self._create_tools()

        with response.create_single_choice() as choice:
            agent = ContentManagementAgent(
                endpoint=DIAL_ENDPOINT,
                tools=self.tools,
            )
            await agent.handle_request(
                choice=choice,
                deployment_name=DEPLOYMENT_NAME,
                request=request,
                response=response,
            )


app: DIALApp = DIALApp()
agent_app = ContentManagementApplication()
app.add_chat_completion("content-management-agent", agent_app)


if __name__ == "__main__":
    uvicorn.run(app, port=5002, host="0.0.0.0")
