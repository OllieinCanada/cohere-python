import base64
import json
from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from cohere.aws_client import map_request_to_bedrock, map_response_from_bedrock


def mapped_events(event: Any, version: int) -> list[Any]:
    payload = base64.b64encode(json.dumps(event).encode()).decode()
    frame = (json.dumps({"bytes": payload}) + "\n").encode()
    request = httpx.Request("POST", "https://example.test/v2/chat")
    request.extensions.update(endpoint="chat", api_version=version)
    response = httpx.Response(
        200,
        request=request,
        headers={"content-type": "application/vnd.amazon.eventstream"},
        stream=httpx.ByteStream(frame),
    )
    map_response_from_bedrock()(response)
    return [json.loads(line) for line in response.read().splitlines()]


@pytest.mark.parametrize(
    "event",
    [
        {"type": "message-start", "id": "message-1", "delta": {"message": {"role": "assistant"}}},
        {"type": "content-delta", "index": 0, "delta": {"message": {"content": {"text": "hello"}}}},
    ],
)
def test_v2_chat_events_keep_their_discriminator_and_payload(event: Any) -> None:
    events = mapped_events(event, 2)

    assert len(events) == 1
    assert events[0]["type"] == event["type"]
    message = events[0]["delta"]["message"]
    if event["type"] == "content-delta":
        assert message["content"]["text"] == "hello"
    else:
        assert message["role"] == "assistant"
    assert "event_type" not in events[0]


def test_v1_chat_events_still_work() -> None:
    events = mapped_events({"event_type": "text-generation", "text": "hello", "is_finished": False}, 1)

    assert len(events) == 1
    assert events[0]["event_type"] == "text-generation"
    assert events[0]["text"] == "hello"


def test_aws_request_hook_preserves_api_version(monkeypatch: Any) -> None:
    session = Mock()
    session.region_name = "us-east-1"
    boto = Mock()
    boto.Session.return_value = session
    botocore = Mock()
    botocore.awsrequest.AWSRequest.return_value.prepare.return_value.headers = {"host": "example.test"}
    monkeypatch.setattr("cohere.aws_client.lazy_boto3", lambda: boto)
    monkeypatch.setattr("cohere.aws_client.lazy_botocore", lambda: botocore)
    request = httpx.Request(
        "POST",
        "https://api.cohere.com/v2/chat",
        headers={"connection": "keep-alive"},
        json={"model": "cohere-model", "stream": True, "messages": []},
    )

    map_request_to_bedrock("bedrock", aws_region="us-east-1")(request)

    assert request.extensions["api_version"] == 2
    assert request.extensions["endpoint"] == "chat"
