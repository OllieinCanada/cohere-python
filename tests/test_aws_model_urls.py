from urllib.parse import unquote, urlsplit

import pytest

from cohere.aws_client import get_url


@pytest.mark.parametrize("stream", [False, True])
def test_bedrock_model_arn_is_one_uri_segment(stream: bool) -> None:
    arn = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc123"
    url = get_url(platform="bedrock", aws_region="us-east-1", model=arn, stream=stream)
    path = urlsplit(url).path
    endpoint = "invoke-with-response-stream" if stream else "invoke"

    assert len(path.split("/")) == 4
    assert unquote(path.split("/")[2]) == arn
    assert path.endswith("/" + endpoint)
    assert "%2F" in path


def test_simple_bedrock_model_is_unchanged() -> None:
    url = get_url(platform="bedrock", aws_region="us-east-1", model="cohere.embed-english-v3", stream=False)

    assert url == "https://bedrock-runtime.us-east-1.amazonaws.com/model/cohere.embed-english-v3/invoke"


def test_sagemaker_endpoint_is_unchanged() -> None:
    url = get_url(platform="sagemaker", aws_region="us-east-1", model="cohere-endpoint", stream=True)

    assert url == (
        "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/cohere-endpoint/invocations-response-stream"
    )
