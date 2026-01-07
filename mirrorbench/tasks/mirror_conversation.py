"""Task driver that alternates synthetic user and assistant turns."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from typing import Any, ClassVar

from mirrorbench.cache import get_cache_manager
from mirrorbench.core.config import RunConfig
from mirrorbench.core.constants import REGISTRY_GROUP_MODEL_CLIENTS
from mirrorbench.core.models.episodes import EpisodeArtifact, EpisodeSpec
from mirrorbench.core.models.errors import TaskDriverError
from mirrorbench.core.models.messages import Message, Role, TurnTelemetry
from mirrorbench.core.models.plan import DatasetSpec
from mirrorbench.core.models.registry import ModelClientInfo, TaskDriverInfo
from mirrorbench.core.registry import registry
from mirrorbench.core.registry.decorators import register_task_driver
from mirrorbench.io.paths import Paths
from mirrorbench.model_clients.base import ChatClient
from mirrorbench.model_clients.caching_wrapper import CachingChatClient
from mirrorbench.model_clients.telemetry import usage_to_turn_telemetry
from mirrorbench.tasks.base import EpisodeExecutionResult, TaskDriver
from mirrorbench.tasks.prompting import (
    build_assistant_mirror_system_prompt,
    build_user_proxy_system_prompt,
)

_MIRROR_DRIVER_NAME = "task:mirror/conversation"
MIRROR_TASK_DRIVER_NAME = _MIRROR_DRIVER_NAME


@register_task_driver(name=MIRROR_TASK_DRIVER_NAME)
class MirrorConversationDriver(TaskDriver):
    """Driver that alternates between synthetic user and assistant generations."""

    info: ClassVar[TaskDriverInfo] = TaskDriverInfo(
        name=_MIRROR_DRIVER_NAME,
        supported_tasks={"mirror_conversation"},
        description="Generate synthetic user turns and assistant replies for mirror datasets.",
    )

    def __init__(self) -> None:
        super().__init__()
        self._assistant_client: ChatClient | None = None
        self._assistant_request_params: dict[str, Any] = {}
        self._system_prompt_override: str | None = None
        self._assistant_system_prompt_override: str | None = None
        self._dataset_name: str | None = None
        self._dataset_label: str | None = None
        self._assistant_model_client_name: str | None = None
        self._assistant_client_params: dict[str, Any] = {}

    def setup(
        self,
        *,
        run_id: str,
        run_config: RunConfig,
        paths: Paths,
        dataset: DatasetSpec,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        super().setup(
            run_id=run_id,
            run_config=run_config,
            paths=paths,
            dataset=dataset,
            params=params,
        )
        params = params or {}
        assistant_name = params.get("assistant_model_client")
        if not isinstance(assistant_name, str) or not assistant_name:
            raise TaskDriverError(
                f"Task driver '{self.info.name}' requires 'assistant_model_client' in params"
            )
        client_params_mapping = params.get("client_params") or {}
        request_params_mapping = params.get("request_params") or {}
        if not isinstance(client_params_mapping, Mapping) or not isinstance(
            request_params_mapping, Mapping
        ):
            raise TaskDriverError(
                "'client_params' and 'request_params' must be mappings when provided"
            )

        client_params = dict(client_params_mapping)
        request_params = dict(request_params_mapping)

        self._system_prompt_override = None
        self._assistant_system_prompt_override = None
        if "system_prompt" in request_params:
            value = request_params.pop("system_prompt")
            if isinstance(value, str) and value.strip():
                prompt_value = value.strip()
                self._system_prompt_override = prompt_value
                self._assistant_system_prompt_override = prompt_value

        factory = registry.factory(REGISTRY_GROUP_MODEL_CLIENTS, assistant_name)
        assistant_client = factory(**client_params)
        if getattr(assistant_client, "info", None) is None:
            assistant_client.info = ModelClientInfo(
                name=assistant_name, provider="unknown", capabilities={"chat"}
            )

        cache_config = run_config.cache
        if cache_config.enabled:
            manager = get_cache_manager(paths, cache_config)
            if manager.enabled:
                namespace = f"assistant:{assistant_name}"
                assistant_client = CachingChatClient(
                    delegate=assistant_client,
                    manager=manager,
                    namespace=namespace,
                    ttl_seconds=cache_config.ttl_seconds,
                )

        self._assistant_client = assistant_client
        self._assistant_request_params = dict(request_params)
        self._assistant_model_client_name = assistant_name
        self._assistant_client_params = dict(client_params_mapping)
        self._dataset_name = dataset.name
        self._dataset_label = dataset.label

    def run_episode(
        self,
        *,
        episode: EpisodeSpec,
        proxy_session: Any,
        run_id: str,
    ) -> EpisodeExecutionResult:
        self._ensure_initialised()
        if self._assistant_client is None:
            raise TaskDriverError("Assistant client has not been configured")

        transcript: list[Message] = []
        artifact_turns: list[Message] = []
        telemetry: list[TurnTelemetry] = []
        previous_role: Role | None = None
        user_turn_index = 0
        first_non_system_role = next(
            (turn.role for turn in episode.chat_history if turn.role is not Role.SYSTEM),
            None,
        )
        seed_starter_assistant = first_non_system_role is Role.ASSISTANT
        assistant_seeded = False

        system_prompt = self._resolve_system_prompt(episode)
        system_message = Message(role=Role.SYSTEM, content=system_prompt) if system_prompt else None

        assistant_prefix: list[Message] = []
        assistant_system_prompt = self._resolve_assistant_system_prompt(episode)
        if assistant_system_prompt:
            assistant_prefix.append(Message(role=Role.SYSTEM, content=assistant_system_prompt))

        for turn in episode.chat_history:
            if turn.role is Role.SYSTEM:
                # Dataset-provided system messages describe the real conversation;
                # we rely on our own system prompt for synthetic orchestration.
                continue

            if previous_role is not None and turn.role is previous_role:
                raise TaskDriverError(
                    "Dataset conversation does not alternate between assistant and user roles"
                )
            previous_role = turn.role

            if turn.role is Role.ASSISTANT:
                if seed_starter_assistant and not assistant_seeded:
                    transcript.append(turn)
                    artifact_turns.append(turn)
                    assistant_seeded = True
                    continue

                assistant_seeded = True
                assistant_message, turn_telemetry = self._generate_assistant_response(
                    transcript,
                    prefix=assistant_prefix,
                )
                transcript.append(assistant_message)
                artifact_turns.append(assistant_message)
                if turn_telemetry is not None:
                    telemetry.append(turn_telemetry)
                continue

            if turn.role is not Role.USER:
                raise TaskDriverError(
                    f"Unsupported role '{turn.role.value}' encountered in episode {episode.episode_id}"
                )

            user_result = self._generate_user_response(
                proxy_session=proxy_session,
                transcript=transcript,
                episode=episode,
                turn_index=user_turn_index,
                system_message=system_message,
            )
            transcript.append(user_result.message)
            artifact_turns.append(user_result.message)
            if user_result.telemetry is not None:
                telemetry.append(user_result.telemetry)
            user_turn_index += 1

        artifact = EpisodeArtifact(spec=episode, turns=artifact_turns, telemetry=list(telemetry))
        return EpisodeExecutionResult(artifact=artifact, turn_telemetries=list(telemetry))

    async def run_episode_async(
        self,
        *,
        episode: EpisodeSpec,
        proxy_session: Any,
        run_id: str,
    ) -> EpisodeExecutionResult:
        self._ensure_initialised()
        if self._assistant_client is None:
            raise TaskDriverError("Assistant client has not been configured")

        transcript: list[Message] = []
        artifact_turns: list[Message] = []
        telemetry: list[TurnTelemetry] = []
        previous_role: Role | None = None
        user_turn_index = 0
        first_non_system_role = next(
            (turn.role for turn in episode.chat_history if turn.role is not Role.SYSTEM),
            None,
        )
        seed_starter_assistant = first_non_system_role is Role.ASSISTANT
        assistant_seeded = False

        system_prompt = self._resolve_system_prompt(episode)
        system_message = Message(role=Role.SYSTEM, content=system_prompt) if system_prompt else None

        assistant_prefix: list[Message] = []
        assistant_system_prompt = self._resolve_assistant_system_prompt(episode)
        if assistant_system_prompt:
            assistant_prefix.append(Message(role=Role.SYSTEM, content=assistant_system_prompt))

        for turn in episode.chat_history:
            if turn.role is Role.SYSTEM:
                continue

            if previous_role is not None and turn.role is previous_role:
                raise TaskDriverError(
                    "Dataset conversation does not alternate between assistant and user roles"
                )
            previous_role = turn.role

            if turn.role is Role.ASSISTANT:
                if seed_starter_assistant and not assistant_seeded:
                    transcript.append(turn)
                    artifact_turns.append(turn)
                    assistant_seeded = True
                    continue

                assistant_seeded = True
                assistant_message, turn_telemetry = await self._generate_assistant_response_async(
                    transcript,
                    prefix=assistant_prefix,
                )
                transcript.append(assistant_message)
                artifact_turns.append(assistant_message)
                if turn_telemetry is not None:
                    telemetry.append(turn_telemetry)
                continue

            if turn.role is not Role.USER:
                raise TaskDriverError(
                    f"Unsupported role '{turn.role.value}' encountered in episode {episode.episode_id}"
                )

            user_result = await self._generate_user_response_async(
                proxy_session=proxy_session,
                transcript=transcript,
                episode=episode,
                turn_index=user_turn_index,
                system_message=system_message,
            )
            transcript.append(user_result.message)
            artifact_turns.append(user_result.message)
            if user_result.telemetry is not None:
                telemetry.append(user_result.telemetry)
            user_turn_index += 1

        artifact = EpisodeArtifact(spec=episode, turns=artifact_turns, telemetry=list(telemetry))
        return EpisodeExecutionResult(artifact=artifact, turn_telemetries=list(telemetry))

    def _resolve_system_prompt(self, episode: EpisodeSpec) -> str | None:
        if getattr(self, "_system_prompt_override", None):
            return self._system_prompt_override

        dataset_name = self._dataset_name or str(episode.metadata.get("dataset", "unknown"))
        return build_user_proxy_system_prompt(
            episode=episode,
            dataset_name=dataset_name,
            dataset_label=self._dataset_label,
        )

    def _resolve_assistant_system_prompt(self, episode: EpisodeSpec) -> str | None:
        if self._assistant_system_prompt_override:
            return self._assistant_system_prompt_override

        dataset_name = self._dataset_name or str(episode.metadata.get("dataset", "unknown"))
        return build_assistant_mirror_system_prompt(
            episode=episode,
            dataset_name=dataset_name,
            dataset_label=self._dataset_label,
        )

    def shutdown(self) -> None:
        if self._assistant_client is not None:
            shutdown = getattr(self._assistant_client, "shutdown", None)
            if callable(shutdown):
                shutdown()
        self._assistant_client = None
        super().shutdown()

    def _generate_user_response(
        self,
        *,
        proxy_session: Any,
        transcript: list[Message],
        episode: EpisodeSpec,
        turn_index: int,
        system_message: Message | None,
    ) -> Any:
        context = list(transcript)
        if system_message is not None:
            context = [system_message, *context]

        turn_spec = EpisodeSpec(
            episode_id=f"{episode.episode_id}:user-{turn_index}",
            task_tag=episode.task_tag,
            chat_history=context,
            references=episode.references,
            metadata=episode.metadata,
            reference_stats=episode.reference_stats,
        )
        result = proxy_session.generate(turn=turn_spec)
        if result.message.role is not Role.USER:
            result_message = Message(
                role=Role.USER,
                content=result.message.content,
                metadata=dict(result.message.metadata),
                name=result.message.name,
                timestamp=result.message.timestamp,
            )
            result.message = result_message
        return result

    async def _generate_user_response_async(
        self,
        *,
        proxy_session: Any,
        transcript: list[Message],
        episode: EpisodeSpec,
        turn_index: int,
        system_message: Message | None,
    ) -> Any:
        context = list(transcript)
        if system_message is not None:
            context = [system_message, *context]

        turn_spec = EpisodeSpec(
            episode_id=f"{episode.episode_id}:user-{turn_index}",
            task_tag=episode.task_tag,
            chat_history=context,
            references=episode.references,
            metadata=episode.metadata,
            reference_stats=episode.reference_stats,
        )

        async_generate = getattr(proxy_session, "generate_async", None)
        if callable(async_generate):
            response = async_generate(turn=turn_spec)
            if inspect.isawaitable(response):
                result = await response
            else:
                result = response
        else:
            result = await asyncio.to_thread(proxy_session.generate, turn=turn_spec)

        if result.message.role is not Role.USER:
            result_message = Message(
                role=Role.USER,
                content=result.message.content,
                metadata=dict(result.message.metadata),
                name=result.message.name,
                timestamp=result.message.timestamp,
            )
            result.message = result_message
        return result

    def _generate_assistant_response(  # noqa: PLR0912
        self,
        transcript: list[Message],
        *,
        prefix: list[Message] | None = None,
    ) -> tuple[Message, TurnTelemetry | None]:
        if self._assistant_client is None:
            raise TaskDriverError("Assistant client has not been configured")
        messages: list[Message] = []
        if prefix:
            messages.extend(prefix)
        messages.extend(transcript)
        response = self._assistant_client.invoke(
            messages=messages,
            **self._assistant_request_params,
        )
        message = response.message
        if message.role is not Role.ASSISTANT:
            message = Message(
                role=Role.ASSISTANT, content=message.content, metadata=message.metadata
            )
        telemetry = usage_to_turn_telemetry(
            response.usage, provider=self._assistant_client.info.provider
        )
        # Tag telemetry with component source and assistant details
        if telemetry is not None:
            if telemetry.metadata is None:
                telemetry.metadata = {}
            telemetry.metadata["component"] = "assistant_proxy"

            # Add assistant model details for rich telemetry
            if self._assistant_model_client_name:
                telemetry.metadata["model_client"] = self._assistant_model_client_name

            if self._assistant_client_params:
                # Store model-related params for identification
                if "model" in self._assistant_client_params:
                    telemetry.metadata["model"] = self._assistant_client_params["model"]
                if "model_name" in self._assistant_client_params:
                    telemetry.metadata["model_name"] = self._assistant_client_params["model_name"]
                if "azure_deployment" in self._assistant_client_params:
                    telemetry.metadata["azure_deployment"] = self._assistant_client_params[
                        "azure_deployment"
                    ]

            if self._assistant_request_params:
                # Store key request parameters
                if "temperature" in self._assistant_request_params:
                    telemetry.metadata["temperature"] = self._assistant_request_params[
                        "temperature"
                    ]
                if "max_tokens" in self._assistant_request_params:
                    telemetry.metadata["max_tokens"] = self._assistant_request_params["max_tokens"]

        return message, telemetry

    async def _generate_assistant_response_async(  # noqa: PLR0912
        self,
        transcript: list[Message],
        *,
        prefix: list[Message] | None = None,
    ) -> tuple[Message, TurnTelemetry | None]:
        if self._assistant_client is None:
            raise TaskDriverError("Assistant client has not been configured")
        messages: list[Message] = []
        if prefix:
            messages.extend(prefix)
        messages.extend(transcript)

        async_invoke = getattr(self._assistant_client, "invoke_async", None)
        if callable(async_invoke):
            response = async_invoke(messages=messages, **self._assistant_request_params)
            if inspect.isawaitable(response):
                assistant_response = await response
            else:
                assistant_response = response
        else:
            assistant_response = await asyncio.to_thread(
                self._assistant_client.invoke,
                messages=messages,
                **self._assistant_request_params,
            )

        message = assistant_response.message
        if message.role is not Role.ASSISTANT:
            message = Message(
                role=Role.ASSISTANT, content=message.content, metadata=message.metadata
            )
        telemetry = usage_to_turn_telemetry(
            assistant_response.usage, provider=self._assistant_client.info.provider
        )
        if telemetry is not None:
            if telemetry.metadata is None:
                telemetry.metadata = {}
            telemetry.metadata["component"] = "assistant_proxy"

            if self._assistant_model_client_name:
                telemetry.metadata["model_client"] = self._assistant_model_client_name

            if self._assistant_client_params:
                if "model" in self._assistant_client_params:
                    telemetry.metadata["model"] = self._assistant_client_params["model"]
                if "model_name" in self._assistant_client_params:
                    telemetry.metadata["model_name"] = self._assistant_client_params["model_name"]
                if "azure_deployment" in self._assistant_client_params:
                    telemetry.metadata["azure_deployment"] = self._assistant_client_params[
                        "azure_deployment"
                    ]

            if self._assistant_request_params:
                if "temperature" in self._assistant_request_params:
                    telemetry.metadata["temperature"] = self._assistant_request_params[
                        "temperature"
                    ]
                if "max_tokens" in self._assistant_request_params:
                    telemetry.metadata["max_tokens"] = self._assistant_request_params["max_tokens"]

        return message, telemetry
