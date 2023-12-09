from fastapi.testclient import TestClient
from .api import build_tool, BingAPI
from typing import Tuple

BING_TEST_SEARCH = {
    "webPages": {
        "value": [
            {"url": "a", "name": "test a", "snippet": "page a"},
            {"url": "b", "name": "test b", "snippet": "page b"},
            {"url": "c", "name": "test c", "snippet": "page c"},
        ]
    }
}


class MockBingAPI(BingAPI):
    def __init__(self):
        pass

    def search(self, key_words: str, max_retry: int = 3):
        return BING_TEST_SEARCH

    def load_page(self, url: str, max_retry: int = 3) -> Tuple[bool, str]:
        if url == "a":
            return True, "This is page a"
        elif url == "b":
            return True, "This is page b"
        elif url == "c":
            return True, "This is page c"
        else:
            return (
                False,
                "Timeout for loading this page, Please try to load another one or search again.",
            )


app = build_tool({"debug": True, "bing_api": MockBingAPI()})
client = TestClient(app)


def test_bing():
    # test search top 3
    response = client.get("/search_top3", params={"key_words": "test"})

    output = ""
    for idx, item in enumerate(BING_TEST_SEARCH["webPages"]["value"]):
        output += "page: " + str(idx + 1) + "\n"
        output += "title: " + item["name"] + "\n"
        output += "summary: " + item["snippet"] + "\n"
    assert response.status_code == 200
    assert response.json() == output

    # test load page
    response = client.get("/load_page_index", params={"idx": "1"})
    assert response.status_code == 200
    assert response.json() == "This is page a"

    response = client.get("/load_page_index", params={"idx": "2"})
    assert response.status_code == 200
    assert response.json() == "This is page b"
