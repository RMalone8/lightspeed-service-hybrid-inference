"""ChatOpenAI that keeps the provider's raw usage block on the chunk that carries it."""

from typing import Any, Optional

from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI


class ChatOpenAIRawUsage(ChatOpenAI):
    """ChatOpenAI that stores the raw ``usage`` object in ``response_metadata["raw_usage"]``.

    LangChain keeps only the usage fields it knows about. Providers also report others, such as
    cache-write tokens, which cost and caching analysis needs.
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: Any,
        default_chunk_class: type,
        base_generation_info: Optional[dict[str, Any]] = None,
    ) -> Optional[ChatGenerationChunk]:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        usage = (
            chunk.get("usage")
            if isinstance(chunk, dict)
            else getattr(chunk, "usage", None)
        )
        if generation_chunk is not None and usage:
            raw = usage if isinstance(usage, dict) else usage.model_dump()
            generation_chunk.message.response_metadata["raw_usage"] = raw
        return generation_chunk
