import logging
from typing import Any, cast

import pytest

import cohere
from cohere.base_client import OMIT, AsyncBaseCohere, BaseCohere


@pytest.mark.parametrize("response_format", [None, OMIT])
def test_sync_optional_response_format_is_forwarded(monkeypatch: Any, response_format: Any) -> None:
    calls: list[dict[str, Any]] = []
    result = object()

    def chat(self: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return result

    monkeypatch.setattr(BaseCohere, "chat", chat)
    with cohere.Client(api_key="unused") as client:
        assert client.chat(message="hello", response_format=response_format) is result

    assert calls == [{"message": "hello", "response_format": response_format}]


@pytest.mark.asyncio
@pytest.mark.parametrize("response_format", [None, OMIT])
async def test_async_optional_response_format_is_forwarded(monkeypatch: Any, response_format: Any) -> None:
    calls: list[dict[str, Any]] = []
    result = object()

    async def chat(self: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return result

    monkeypatch.setattr(AsyncBaseCohere, "chat", chat)
    async with cohere.AsyncClient(api_key="unused") as client:
        assert await client.chat(message="hello", response_format=response_format) is result

    assert calls == [{"message": "hello", "response_format": response_format}]


def test_present_schema_still_warns(monkeypatch: Any, caplog: Any) -> None:
    monkeypatch.setattr(BaseCohere, "chat", lambda self, **kwargs: "ok")
    response_format: Any = {"type": "json_object", "schema": {"type": "object"}}
    with caplog.at_level(logging.WARNING), cohere.Client(api_key="unused") as client:
        assert client.chat(message="hello", response_format=response_format) == "ok"

    assert sum("response_format.schema" in r.message for r in caplog.records) == 1


def test_missing_schema_does_not_warn(monkeypatch: Any, caplog: Any) -> None:
    monkeypatch.setattr(BaseCohere, "chat", lambda self, **kwargs: "ok")
    with caplog.at_level(logging.WARNING), cohere.Client(api_key="unused") as client:
        assert client.chat(message="hello", response_format=cast(Any, {"type": "json_object"})) == "ok"

    assert not any("response_format.schema" in r.message for r in caplog.records)
