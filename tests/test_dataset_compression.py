import gzip
import zlib
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast

import pytest
import requests
from fastavro import writer
from urllib3.response import HTTPResponse

from cohere import Dataset
from cohere.utils import dataset_generator


@pytest.mark.parametrize("encoding", ["gzip", "deflate", None])
def test_http_compressed_dataset_parts_are_decoded(monkeypatch: Any, encoding: str | None) -> None:
    buffer = BytesIO()
    schema = {"type": "record", "name": "Row", "fields": [{"name": "text", "type": "string"}]}
    records = [{"text": "first record"}, {"text": "second record"}]
    writer(buffer, schema, records)
    data = buffer.getvalue()
    if encoding == "gzip":
        data = gzip.compress(data)
    elif encoding == "deflate":
        data = zlib.compress(data)
    headers = {"Content-Encoding": encoding} if encoding else {}
    raw = HTTPResponse(body=BytesIO(data), headers=headers, preload_content=False, decode_content=False)
    response = requests.Response()
    response.status_code = 200
    response.raw = raw
    response.headers.update(headers)
    response.url = "https://example.test/dataset.avro"
    monkeypatch.setattr("cohere.utils.requests.get", lambda *args, **kwargs: response)
    dataset = cast(Dataset, SimpleNamespace(dataset_parts=[SimpleNamespace(url=response.url)]))

    assert list(dataset_generator(dataset)) == records
    assert raw.closed
