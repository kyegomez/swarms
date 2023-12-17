import pytest
import requests

from swarms.models.gigabind import Gigabind

try:
    import requests_mock
except ImportError:
    requests_mock = None


@pytest.fixture
def api():
    return Gigabind(
        host="localhost", port=8000, endpoint="embeddings"
    )


@pytest.fixture
def mock(requests_mock):
    requests_mock.post(
        "http://localhost:8000/embeddings", json={"result": "success"}
    )
    return requests_mock


def test_run_with_text(api, mock):
    response = api.run(text="Hello, world!")
    assert response == {"result": "success"}


def test_run_with_vision(api, mock):
    response = api.run(vision="image.jpg")
    assert response == {"result": "success"}


def test_run_with_audio(api, mock):
    response = api.run(audio="audio.mp3")
    assert response == {"result": "success"}


def test_run_with_all(api, mock):
    response = api.run(
        text="Hello, world!", vision="image.jpg", audio="audio.mp3"
    )
    assert response == {"result": "success"}


def test_run_with_none(api):
    with pytest.raises(ValueError):
        api.run()


def test_generate_summary(api, mock):
    response = api.generate_summary(text="Hello, world!")
    assert response == {"result": "success"}


def test_generate_summary_with_none(api):
    with pytest.raises(ValueError):
        api.generate_summary()


def test_retry_on_failure(api, requests_mock):
    requests_mock.post(
        "http://localhost:8000/embeddings",
        [
            {"status_code": 500, "json": {}},
            {"status_code": 500, "json": {}},
            {"status_code": 200, "json": {"result": "success"}},
        ],
    )
    response = api.run(text="Hello, world!")
    assert response == {"result": "success"}


def test_retry_exhausted(api, requests_mock):
    requests_mock.post(
        "http://localhost:8000/embeddings",
        [
            {"status_code": 500, "json": {}},
            {"status_code": 500, "json": {}},
            {"status_code": 500, "json": {}},
        ],
    )
    response = api.run(text="Hello, world!")
    assert response is None


def test_proxy_url(api):
    api.proxy_url = "http://proxy:8080"
    assert api.url == "http://proxy:8080"


def test_invalid_response(api, requests_mock):
    requests_mock.post(
        "http://localhost:8000/embeddings", text="not json"
    )
    response = api.run(text="Hello, world!")
    assert response is None


def test_connection_error(api, requests_mock):
    requests_mock.post(
        "http://localhost:8000/embeddings",
        exc=requests.exceptions.ConnectTimeout,
    )
    response = api.run(text="Hello, world!")
    assert response is None


def test_http_error(api, requests_mock):
    requests_mock.post(
        "http://localhost:8000/embeddings", status_code=500
    )
    response = api.run(text="Hello, world!")
    assert response is None


def test_url_construction(api):
    assert api.url == "http://localhost:8000/embeddings"


def test_url_construction_with_proxy(api):
    api.proxy_url = "http://proxy:8080"
    assert api.url == "http://proxy:8080"


def test_run_with_large_text(api, mock):
    large_text = "Hello, world! " * 10000  # 10,000 repetitions
    response = api.run(text=large_text)
    assert response == {"result": "success"}


def test_run_with_large_vision(api, mock):
    large_vision = "image.jpg" * 10000  # 10,000 repetitions
    response = api.run(vision=large_vision)
    assert response == {"result": "success"}


def test_run_with_large_audio(api, mock):
    large_audio = "audio.mp3" * 10000  # 10,000 repetitions
    response = api.run(audio=large_audio)
    assert response == {"result": "success"}


def test_run_with_large_all(api, mock):
    large_text = "Hello, world! " * 10000  # 10,000 repetitions
    large_vision = "image.jpg" * 10000  # 10,000 repetitions
    large_audio = "audio.mp3" * 10000  # 10,000 repetitions
    response = api.run(
        text=large_text, vision=large_vision, audio=large_audio
    )
    assert response == {"result": "success"}


def test_run_with_timeout(api, mock):
    response = api.run(text="Hello, world!", timeout=0.001)
    assert response is None


def test_run_with_invalid_host(api):
    api.host = "invalid"
    response = api.run(text="Hello, world!")
    assert response is None


def test_run_with_invalid_port(api):
    api.port = 99999
    response = api.run(text="Hello, world!")
    assert response is None


def test_run_with_invalid_endpoint(api):
    api.endpoint = "invalid"
    response = api.run(text="Hello, world!")
    assert response is None


def test_run_with_invalid_proxy_url(api):
    api.proxy_url = "invalid"
    response = api.run(text="Hello, world!")
    assert response is None
