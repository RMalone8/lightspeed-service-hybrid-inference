"""Token usage the provider reported for one tool-calling round."""

from dataclasses import dataclass
from typing import Any, Optional


def _count(value: Any) -> int:
    """Return a usage field as an int, treating missing or null as 0."""
    return int(value) if isinstance(value, (int, float)) else 0


def usage_from_chunk(chunk: Any) -> Optional[dict[str, int]]:
    """Return the usage a streamed chunk carries, or None when it carries none.

    Prefers the raw provider usage block, which includes cache-write tokens, and falls back
    to the usage metadata LangChain builds from it.
    """
    raw = (getattr(chunk, "response_metadata", None) or {}).get("raw_usage")
    if raw:
        prompt_details = raw.get("prompt_tokens_details") or {}
        completion_details = raw.get("completion_tokens_details") or {}
        return {
            "prompt_tokens": _count(raw.get("prompt_tokens")),
            "completion_tokens": _count(raw.get("completion_tokens")),
            "cached_tokens": _count(prompt_details.get("cached_tokens")),
            "cache_write_tokens": _count(prompt_details.get("created_cache_tokens")),
            "reasoning_tokens": _count(completion_details.get("reasoning_tokens")),
        }
    metadata = getattr(chunk, "usage_metadata", None)
    if metadata:
        input_details = metadata.get("input_token_details") or {}
        output_details = metadata.get("output_token_details") or {}
        return {
            "prompt_tokens": _count(metadata.get("input_tokens")),
            "completion_tokens": _count(metadata.get("output_tokens")),
            "cached_tokens": _count(input_details.get("cache_read")),
            "cache_write_tokens": _count(input_details.get("cache_creation")),
            "reasoning_tokens": _count(output_details.get("reasoning")),
        }
    return None


@dataclass
class RoundUsage:
    """Usage the provider reported for one round, summed over its attempts.

    prompt_tokens includes cached and cache-write tokens, and completion_tokens includes
    reasoning tokens, as providers report them.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    model: Optional[str] = None
    reported: bool = False

    def add(self, usage: dict[str, int]) -> None:
        """Add the usage of one attempt."""
        self.prompt_tokens += usage["prompt_tokens"]
        self.completion_tokens += usage["completion_tokens"]
        self.cached_tokens += usage["cached_tokens"]
        self.cache_write_tokens += usage["cache_write_tokens"]
        self.reasoning_tokens += usage["reasoning_tokens"]
        self.reported = True

    def to_record(self, round_index: int, handoff: bool) -> dict[str, Any]:
        """Return this round as a JSON-ready record; token fields are None when not reported."""
        record: dict[str, Any] = {
            "round": round_index,
            "handoff": handoff,
            "model": self.model,
            "usage_reported": self.reported,
        }
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
        ):
            record[name] = getattr(self, name) if self.reported else None
        return record
