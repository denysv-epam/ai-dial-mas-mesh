from typing import Any

from task.tools.deployment.base_agent_tool import BaseAgentTool


class WebSearchAgentTool(BaseAgentTool):

    @property
    def deployment_name(self) -> str:
        """Return the remote deployment name from core config.

        This value must match the `applications` key in `core/config.json`
        for correct routing to the Web Search agent.
        """
        return "web-search-agent"

    @property
    def name(self) -> str:
        """Return the function/tool name exposed to the LLM.

        This identifier is used for tool calls and for persisting
        per-agent history inside assistant state.
        """
        return "web_search_agent"

    @property
    def description(self) -> str:
        """Describe when to call the Web Search Agent.

        Use this tool for web research tasks that require external
        sources or live information.
        """
        return "Routes requests to the Web Search Agent for web research."

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
                    "description": "Request to the web search agent.",
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
