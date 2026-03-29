from typing import Any

from task.tools.deployment.base_agent_tool import BaseAgentTool


class ContentManagementAgentTool(BaseAgentTool):

    @property
    def deployment_name(self) -> str:
        """Return the remote deployment name from core config.

        This must match the `applications` key in `core/config.json` so
        the tool routes requests to the correct agent.
        """
        return "content-management-agent"

    @property
    def name(self) -> str:
        """Return the function/tool name exposed to the LLM.

        This identifier is used for tool calls and for storing per-agent
        call history inside assistant state.
        """
        return "content_management_agent"

    @property
    def description(self) -> str:
        """Describe when to call the Content Management Agent.

        This tool should be used for file extraction, document analysis,
        and retrieval-augmented search across uploaded content.
        """
        return (
            "Routes requests to the Content Management Agent for file "
            "content extraction and RAG search."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Define the tool call schema for agent routing.

        The tool accepts a required `prompt` with the downstream request
        and an optional `propagate_history` flag that controls whether
        per-agent history is reconstructed.
        """
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Request to the content management agent."
                    ),
                },
                "propagate_history": {
                    "type": "boolean",
                    "description": (
                        "Whether to include prior per-agent conversation "
                        "history in this call."
                    ),
                },
            },
            "required": ["prompt"],
        }
