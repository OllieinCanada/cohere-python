from typing import Any

import httpx
import pytest

import cohere
from cohere.aws_client import map_response_from_bedrock
from cohere.core.api_error import ApiError


def client_for(status: int, body: Any, headers: Any = None) -> cohere.Client:
    def respond(request: httpx.Request) -> httpx.Response:
        request.extensions["endpoint"] = "chat"
        if isinstance(body, bytes):
            return httpx.Response(status, content=body, headers=headers)
        return httpx.Response(status, json=body, headers=headers)

    transport = httpx.MockTransport(respond)
    client = httpx.Client(transport=transport, event_hooks={"response": [map_response_from_bedrock()]})
    return cohere.Client(api_key="unused", httpx_client=client)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (403, {"message": "not allowed"}),
        (429, {"message": "rate limited"}),
        (503, b"service unavailable"),
    ],
)
def test_aws_download_errors_reach_sdk_error_surface(status: int, body: Any) -> None:
    with client_for(status, body) as client:
        with pytest.raises(ApiError) as exc_info:
            client.chat(message="hello")

    assert exc_info.value.status_code == status
    expected = body.decode() if isinstance(body, bytes) else body
    assert exc_info.value.body == expected


def test_aws_success_response_still_maps_token_counts() -> None:
    headers = {
        "content-type": "application/json",
        "X-Amzn-Bedrock-Input-Token-Count": "12",
        "X-Amzn-Bedrock-Output-Token-Count": "7",
    }
    with client_for(200, {"text": "hello", "generation_id": "test-generation"}, headers) as client:
        response = client.chat(message="hello")

    assert response.text == "hello"
    assert response.meta is not None
    assert response.meta.tokens is not None
    assert response.meta.tokens.input_tokens == 12
    assert response.meta.tokens.output_tokens == 7
