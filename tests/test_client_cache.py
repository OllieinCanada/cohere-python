from typing import Any

import pytest

import cohere


@pytest.mark.parametrize("client_type", [cohere.Client, cohere.ClientV2])
def test_sync_clients_have_separate_tokenizer_caches(client_type: Any) -> None:
    with (
        client_type(api_key="unused", base_url="https://first.test") as first,
        client_type(api_key="unused", base_url="https://second.test") as second,
    ):
        first._cache_set("tokenizers:shared-model", "first-tokenizer")
        assert second._cache_get("tokenizers:shared-model") is None
        second._cache_set("tokenizers:shared-model", "second-tokenizer")
        assert first._cache_get("tokenizers:shared-model") == "first-tokenizer"
        assert second._cache_get("tokenizers:shared-model") == "second-tokenizer"


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [cohere.AsyncClient, cohere.AsyncClientV2])
async def test_async_clients_have_separate_tokenizer_caches(client_type: Any) -> None:
    async with (
        client_type(api_key="unused", base_url="https://first.test") as first,
        client_type(api_key="unused", base_url="https://second.test") as second,
    ):
        first._cache_set("tokenizers:shared-model", "first-tokenizer")
        assert second._cache_get("tokenizers:shared-model") is None
        second._cache_set("tokenizers:shared-model", "second-tokenizer")
        assert first._cache_get("tokenizers:shared-model") == "first-tokenizer"
        assert second._cache_get("tokenizers:shared-model") == "second-tokenizer"


def test_client_cache_preserves_reuse_and_expiry(monkeypatch: Any) -> None:
    monkeypatch.setattr("cohere.manually_maintained.cache.time.time", lambda: 100.0)
    with cohere.Client(api_key="unused") as client:
        client._cache_set("fresh", "tokenizer", ttl=60)
        assert client._cache_get("fresh") == "tokenizer"
        client._cache_set("expired", "old-tokenizer", ttl=0)
        assert client._cache_get("expired") is None
