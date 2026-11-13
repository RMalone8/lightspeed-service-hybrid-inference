"""Unit tests for provider-reported token usage per round."""

from langchain_core.messages.ai import AIMessageChunk

from ols.src.query_helpers.round_usage import RoundUsage, usage_from_chunk

RAW = {
    "prompt_tokens": 3935,
    "completion_tokens": 19,
    "completion_tokens_details": {"reasoning_tokens": 15},
    "prompt_tokens_details": {"cached_tokens": 3904, "created_cache_tokens": 12},
}


def test_usage_from_raw_usage_includes_cache_writes_and_reasoning() -> None:
    """The raw provider block gives cache reads, cache writes and reasoning tokens."""
    chunk = AIMessageChunk(content="", response_metadata={"raw_usage": RAW})
    assert usage_from_chunk(chunk) == {
        "prompt_tokens": 3935,
        "completion_tokens": 19,
        "cached_tokens": 3904,
        "cache_write_tokens": 12,
        "reasoning_tokens": 15,
    }


def test_usage_falls_back_to_langchain_usage_metadata() -> None:
    """Without the raw block, LangChain's usage metadata is used (no cache writes)."""
    chunk = AIMessageChunk(
        content="",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_token_details": {"cache_read": 60},
            "output_token_details": {"reasoning": 5},
        },
    )
    assert usage_from_chunk(chunk) == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cached_tokens": 60,
        "cache_write_tokens": 0,
        "reasoning_tokens": 5,
    }


def test_null_and_missing_detail_fields_count_as_zero() -> None:
    """A provider may send null or omit detail fields."""
    raw = {
        "prompt_tokens": 50,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": None, "multimodal_tokens": None},
    }
    chunk = AIMessageChunk(content="", response_metadata={"raw_usage": raw})
    assert usage_from_chunk(chunk) == {
        "prompt_tokens": 50,
        "completion_tokens": 5,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }


def test_chunk_without_usage_returns_none() -> None:
    """Text chunks carry no usage."""
    assert usage_from_chunk(AIMessageChunk(content="hello")) is None


def test_round_usage_sums_attempts() -> None:
    """A retried round adds the usage of every attempt."""
    usage = RoundUsage()
    attempt = {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "cached_tokens": 4,
        "cache_write_tokens": 1,
        "reasoning_tokens": 1,
    }
    usage.add(attempt)
    usage.add(attempt)
    usage.model = "m"
    assert usage.to_record(2, True) == {
        "round": 2,
        "handoff": True,
        "model": "m",
        "usage_reported": True,
        "prompt_tokens": 20,
        "completion_tokens": 4,
        "cached_tokens": 8,
        "cache_write_tokens": 2,
        "reasoning_tokens": 2,
    }


def test_unreported_round_has_no_token_numbers() -> None:
    """A round whose provider reported nothing is marked, not estimated."""
    assert RoundUsage().to_record(1, False) == {
        "round": 1,
        "handoff": False,
        "model": None,
        "usage_reported": False,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
        "cache_write_tokens": None,
        "reasoning_tokens": None,
    }
