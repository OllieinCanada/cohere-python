from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import requests

from cohere.manually_maintained import tokenizers


def setup_download(monkeypatch: Any, async_mode: bool, status: int = 200) -> tuple[Any, ...]:
    co: Any = Mock()
    co._cache_get.return_value = None
    model = SimpleNamespace(tokenizer_url="https://example.test/tokenizer.json")
    co.models.get = AsyncMock(return_value=model) if async_mode else Mock(return_value=model)
    response = requests.Response()
    response.status_code = status
    response.url = model.tokenizer_url
    response._content = b'{"version":"test"}'
    request = Mock(return_value=response)
    marker = object()
    decoder = Mock(return_value=marker)
    monkeypatch.setattr(tokenizers.requests, "get", request)
    monkeypatch.setattr(tokenizers, "_get_tokenizer_config_size", lambda url: 1.0)
    monkeypatch.setattr(tokenizers, "Tokenizer", SimpleNamespace(from_str=decoder))
    return co, response, request, decoder, marker


async def download(co: Any, async_mode: bool) -> Any:
    if async_mode:
        return await tokenizers.async_get_hf_tokenizer(co, "embed-model")
    return tokenizers.get_hf_tokenizer(co, "embed-model")


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("status", [403, 503])
async def test_tokenizer_download_preserves_http_error(monkeypatch: Any, async_mode: bool, status: int) -> None:
    co, response, request, decoder, marker = setup_download(monkeypatch, async_mode, status)
    decoder.side_effect = ValueError("invalid tokenizer document")

    with pytest.raises(requests.HTTPError) as exc_info:
        await download(co, async_mode)

    assert exc_info.value.response is response
    assert exc_info.value.response.status_code == status
    decoder.assert_not_called()
    co._cache_set.assert_not_called()
    request.assert_called_once_with(response.url)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_successful_tokenizer_download_is_cached(monkeypatch: Any, async_mode: bool) -> None:
    co, response, request, decoder, marker = setup_download(monkeypatch, async_mode)

    assert await download(co, async_mode) is marker
    decoder.assert_called_once_with(response.text)
    co._cache_set.assert_called_once_with(tokenizers.tokenizer_cache_key("embed-model"), marker)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_cached_tokenizer_does_not_download(monkeypatch: Any, async_mode: bool) -> None:
    co, response, request, decoder, marker = setup_download(monkeypatch, async_mode)
    co._cache_get.return_value = marker

    assert await download(co, async_mode) is marker
    co.models.get.assert_not_called()
    request.assert_not_called()
    decoder.assert_not_called()
