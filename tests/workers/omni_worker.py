import pytest

from swarms.worker.omni_worker import OmniWorkerAgent


@pytest.fixture
def omni_worker():
    api_key = "test-key"
    api_endpoint = "test-endpoint"
    api_type = "test-type"
    return OmniWorkerAgent(api_key, api_endpoint, api_type)


@pytest.mark.parametrize(
    "data, expected_response",
    [
        (
            {
                "messages": ["Hello"],
                "api_key": "key1",
                "api_type": "type1",
                "api_endpoint": "endpoint1",
            },
            {"response": "Hello back from Huggingface!"},
        ),
        (
            {
                "messages": ["Goodbye"],
                "api_key": "key2",
                "api_type": "type2",
                "api_endpoint": "endpoint2",
            },
            {"response": "Goodbye from Huggingface!"},
        ),
    ],
)
def test_chat_valid_data(mocker, omni_worker, data, expected_response):
    mocker.patch(
        "yourmodule.chat_huggingface", return_value=expected_response
    )  # replace 'yourmodule' with actual module name
    assert omni_worker.chat(data) == expected_response


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"messages": ["Hello"]},  # missing api_key, api_type and api_endpoint
        {"messages": ["Hello"], "api_key": "key1"},  # missing api_type and api_endpoint
        {
            "messages": ["Hello"],
            "api_key": "key1",
            "api_type": "type1",
        },  # missing api_endpoint
    ],
)
def test_chat_invalid_data(omni_worker, invalid_data):
    with pytest.raises(ValueError):
        omni_worker.chat(invalid_data)
