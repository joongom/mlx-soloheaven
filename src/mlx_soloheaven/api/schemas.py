"""OpenAI-compatible API schemas."""

import time
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# --- Request schemas ---

class FunctionDef(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = {}


class ToolDef(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


class MessageToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: dict[str, str]  # {"name": ..., "arguments": ...}


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: Optional[str | list] = None
    name: Optional[str] = None
    tool_calls: Optional[list[MessageToolCall]] = None
    tool_call_id: Optional[str] = None


# --- Structured output (response_format) ---

class JsonSchemaSpec(BaseModel):
    """OpenAI response_format.json_schema content."""
    name: Optional[str] = None
    description: Optional[str] = None
    schema_: Optional[dict[str, Any]] = Field(default=None, alias="schema")
    strict: Optional[bool] = None
    model_config = {"populate_by_name": True}


class ResponseFormat(BaseModel):
    """OpenAI response_format parameter.

    Values:
    - {"type": "text"}: no constraint (default)
    - {"type": "json_object"}: loose JSON mode (any valid JSON object)
    - {"type": "json_schema", "json_schema": {name, schema, strict}}: strict schema
    """
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[JsonSchemaSpec] = None


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[str | list[str]] = None
    tools: Optional[list[ToolDef]] = None
    tool_choice: Optional[str | dict] = None
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    thinking: Optional[bool] = None  # Enable/disable thinking (default: server config)
    thinking_budget: Optional[int] = None  # Override thinking token budget
    # Extended sampling parameters
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None  # OpenAI compat (mapped to repetition_penalty)
    presence_penalty: Optional[float] = None   # OpenAI compat (mapped to repetition_penalty)


# --- Response schemas ---

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_info: Optional[dict] = None


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: list[Choice]
    usage: UsageInfo = UsageInfo()


# --- Streaming schemas ---

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: list[ChunkChoice]
    usage: Optional[UsageInfo] = None


# --- Model listing ---

class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mlx-soloheaven"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
