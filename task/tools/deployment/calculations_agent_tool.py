from typing import Any

from task.tools.deployment.base_agent_tool import BaseAgentTool


class CalculationsAgentTool(BaseAgentTool):

    @property
    def deployment_name(self) -> str:
        """Return the remote deployment name from core config.

        This value must match the `applications` entry in `core/config.json`
        so the tool can route requests to the correct agent.
        """
        return "calculations-agent"

    @property
    def name(self) -> str:
        """Return the function/tool name exposed to the LLM.

        This name is used as the tool identifier and also as the key for
        storing per-tool call history in assistant state.
        """
        return "calculations_agent"

    @property
    def description(self) -> str:
        """Describe when to call the Calculations Agent.

        The description should steer the model toward this tool for
        numeric reasoning, analysis, and charting tasks.
        """
        return (
            "Routes requests to the Calculations Agent for math, data "
            "analysis, and plotting tasks."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Define the tool call schema for agent routing.

        The tool accepts a required `prompt` with the request to the
        downstream agent and an optional `propagate_history` flag that
        determines whether per-agent history is reconstructed.
        """
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Request to the calculations agent.",
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
