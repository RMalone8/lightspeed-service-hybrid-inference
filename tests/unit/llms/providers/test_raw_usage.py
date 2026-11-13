"""Unit tests for the ChatOpenAI client that keeps the raw usage block."""

from typing import Any, Optional

from langchain_core.messages import AIMessageChunk

from ols.src.llms.providers.raw_usage import ChatOpenAIRawUsage

DEEPSEEK_STYLE_USAGE = {
    "prompt_tokens": 3935,
    "completion_tokens": 19,
    "total_tokens": 3954,
    "completion_tokens_details": {"reasoning_tokens": 15},
    "prompt_tokens_details": {"cached_tokens": 3904, "created_cache_tokens": 12},
}


def _chunk(usage: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Build a streamed chunk; the usage chunk has no choices, as providers send it."""
    chunk: dict[str, Any] = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": (
            []
            if usage
            else [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]
        ),
    }
    if usage:
        chunk["usage"] = usage
    return chunk


def test_raw_usage_is_kept_with_the_fields_langchain_drops() -> None:
    """Cache-write tokens survive in response_metadata next to LangChain's usage metadata."""
    llm = ChatOpenAIRawUsage(api_key="key", model="m")
    generation = llm._convert_chunk_to_generation_chunk(
        _chunk(DEEPSEEK_STYLE_USAGE), AIMessageChunk, None
    )
    assert generation is not None
    assert generation.message.response_metadata["raw_usage"] == DEEPSEEK_STYLE_USAGE
    assert (
        generation.message.usage_metadata["input_token_details"]["cache_read"] == 3904
    )


def test_chunks_without_usage_get_no_raw_usage() -> None:
    """Text chunks carry no raw usage."""
    llm = ChatOpenAIRawUsage(api_key="key", model="m")
    generation = llm._convert_chunk_to_generation_chunk(_chunk(), AIMessageChunk, None)
    assert generation is not None
    assert "raw_usage" not in generation.message.response_metadata
