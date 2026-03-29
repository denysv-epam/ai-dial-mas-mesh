import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent, Stage, Attachment
from pydantic import StrictStr

from task.tools.base_tool import BaseTool
from task.tools.models import ToolCallParams
from task.utils.stage import StageProcessor


class BaseAgentTool(BaseTool, ABC):

    def __init__(self, endpoint: str):
        """Initialize the tool with the target agent endpoint.

        Args:
            endpoint: Base URL where the agent exposes `/chat/completions`.
        """
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        """Return the deployment name configured in core config."""
        pass

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        """Invoke a remote agent through DIAL and stream its output.

        The tool calls another agent via the unified `/chat/completions` interface,
        streams back content into the current tool stage, and propagates custom
        content (attachments, state, and nested stages) into the current choice.

        Steps:
        1. Validate required `prompt` argument.
        2. Use AsyncDial with streaming enabled and pass `x-conversation-id`.
        3. Collect streamed text, custom content, and propagated stages.
        4. Forward attachments and stage updates into the current choice.
        5. Close any opened propagated stages safely.
        6. Return a tool message with content and custom content state.
        """
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        prompt = arguments.get("prompt")
        if not prompt:
            return Message(
                role=Role.TOOL,
                name=StrictStr(tool_call_params.tool_call.function.name),
                tool_call_id=StrictStr(tool_call_params.tool_call.id),
                content=StrictStr("Error: prompt is required."),
            )

        client: AsyncDial = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version="2025-01-01-preview",
        )

        extra_headers = None
        if tool_call_params.conversation_id:
            extra_headers = {
                "x-conversation-id": tool_call_params.conversation_id
            }

        chunks = await client.chat.completions.create(
            messages=self._prepare_messages(tool_call_params),
            stream=True,
            deployment_name=self.deployment_name,
            extra_headers=extra_headers,
        )

        content = ""
        custom_content: CustomContent = CustomContent(attachments=[])
        stages_map: dict[int, Stage] = {}

        async for chunk in chunks:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if not delta:
                continue

            if delta.content:
                if tool_call_params.stage:
                    tool_call_params.stage.append_content(delta.content)
                content += delta.content

            if delta.custom_content:
                response_custom_content = delta.custom_content

                if response_custom_content.state is not None:
                    if custom_content.state is None:
                        custom_content.state = {}
                    if isinstance(response_custom_content.state, dict):
                        custom_content.state.update(response_custom_content.state)
                    else:
                        custom_content.state = response_custom_content.state

                if response_custom_content.attachments:
                    custom_content.attachments.extend(
                        response_custom_content.attachments
                    )
                    for attachment in response_custom_content.attachments:
                        if isinstance(attachment, Attachment):
                            tool_call_params.choice.add_attachment(attachment)
                            continue

                        attachment_data: dict[str, Any]
                        if hasattr(attachment, "model_dump"):
                            attachment_data = attachment.model_dump(exclude_none=True)
                        elif hasattr(attachment, "dict"):
                            attachment_data = attachment.dict(exclude_none=True)
                        elif isinstance(attachment, dict):
                            attachment_data = {
                                key: value
                                for key, value in attachment.items()
                                if value is not None
                            }
                        else:
                            attachment_data = {
                                "type": getattr(attachment, "type", None),
                                "title": getattr(attachment, "title", None),
                                "data": getattr(attachment, "data", None),
                                "url": getattr(attachment, "url", None),
                                "reference_type": getattr(attachment, "reference_type", None),
                                "reference_url": getattr(attachment, "reference_url", None),
                            }
                            attachment_data = {
                                key: value
                                for key, value in attachment_data.items()
                                if value is not None
                            }
                        try:
                            tool_call_params.choice.add_attachment(
                                Attachment(**attachment_data)
                            )
                        except Exception:
                            continue

                response_custom_content_dict = response_custom_content.dict(
                    exclude_none=True
                )
                stages_data = response_custom_content_dict.get("stages")
                if stages_data:
                    for stage_data in stages_data:
                        stage_index = stage_data.get("index")
                        if stage_index is None:
                            continue

                        stage = stages_map.get(stage_index)
                        if not stage:
                            stage = StageProcessor.open_stage(
                                choice=tool_call_params.choice,
                                name=stage_data.get("name"),
                            )
                            stages_map[stage_index] = stage

                        if stage_data.get("content"):
                            stage.append_content(stage_data["content"])

                        stage_attachments = stage_data.get("attachments") or []
                        for attachment in stage_attachments:
                            stage.add_attachment(
                                type=attachment.get("type"),
                                title=attachment.get("title"),
                                data=attachment.get("data"),
                                url=attachment.get("url"),
                                reference_url=attachment.get("reference_url"),
                                reference_type=attachment.get("reference_type"),
                            )

                        if stage_data.get("status") == "completed":
                            StageProcessor.close_stage_safely(stage)

        for stage in stages_map.values():
            StageProcessor.close_stage_safely(stage)

        return Message(
            role=Role.TOOL,
            name=StrictStr(tool_call_params.tool_call.function.name),
            content=StrictStr(content),
            custom_content=custom_content,
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
        )

    def _prepare_messages(self, tool_call_params: ToolCallParams) -> list[dict[str, Any]]:
        """Build the message history for the downstream agent call.

        The tool supports two modes:
        - One-shot: only the latest user prompt is sent.
        - Propagate history: reconstruct per-agent tool call history stored
          in assistant message state under `self.name` and include the
          preceding user message for each assistant message.

        The final user message includes the prompt and forwards any
        custom_content from the original last message (attachments/urls).
        """
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        prompt = arguments.get("prompt")
        propagate_history = arguments.get("propagate_history", False)

        messages: list[dict[str, Any]] = []

        if propagate_history:
            for idx, message in enumerate(tool_call_params.messages):
                if message.role != Role.ASSISTANT:
                    continue

                custom_content = message.custom_content
                if not custom_content or not custom_content.state:
                    continue

                state = custom_content.state
                if not isinstance(state, dict):
                    continue

                agent_history = state.get(self.name)
                if not agent_history:
                    continue

                if idx > 0:
                    messages.append(
                        tool_call_params.messages[idx - 1].dict(exclude_none=True)
                    )

                restored_message = deepcopy(message)
                restored_message.custom_content.state = agent_history
                messages.append(restored_message.dict(exclude_none=True))

        user_message: dict[str, Any] = {
            "role": Role.USER.value,
            "content": prompt or "",
        }

        last_message_custom_content = None
        if tool_call_params.messages:
            last_message = tool_call_params.messages[-1]
            if last_message.custom_content:
                last_message_custom_content = deepcopy(last_message.custom_content)

        if last_message_custom_content:
            user_message["custom_content"] = last_message_custom_content.dict(
                exclude_none=True
            )

        messages.append(user_message)

        return messages
