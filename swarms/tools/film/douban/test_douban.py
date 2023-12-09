from fastapi.testclient import TestClient
from .api import build_tool, DoubanAPI, ComingMovieInfo, PlayingMovieInfo
from typing import List


class DoubanMock(DoubanAPI):
    def __init__(self) -> None:
        pass

    def get_coming(self) -> List[ComingMovieInfo]:
        return [
            ComingMovieInfo(
                date="2020-12-12",
                title="test1",
                cate="test1",
                region="test1",
                wantWatchPeopleNum="1",
                link="test1",
            ),
            ComingMovieInfo(
                date="2021-12-12",
                title="test2",
                cate="test2",
                region="test2",
                wantWatchPeopleNum="2",
                link="test2",
            ),
            ComingMovieInfo(
                date="2022-12-12",
                title="test3",
                cate="test3",
                region="test3",
                wantWatchPeopleNum="3",
                link="test3",
            ),
        ]

    def get_now_playing(self) -> List[PlayingMovieInfo]:
        return [
            PlayingMovieInfo(
                title="test1",
                score="1.1",
                region="test1",
                director="test1",
                actors="test1",
                link="test1",
            ),
            PlayingMovieInfo(
                title="test2",
                score="2.2",
                region="test2",
                director="test2",
                actors="test2",
                link="test2",
            ),
            PlayingMovieInfo(
                title="test3",
                score="3.3",
                region="test3",
                director="test3",
                actors="test3",
                link="test3",
            ),
        ]

    def get_movie_detail(self, url: str) -> str:
        return url


app = build_tool({"debug": True, "douban_api": DoubanMock()})
client = TestClient(app)


def test_get_coming():
    response = client.get("/coming_out_filter", params={"args": "全部, 全部, 2, True"})
    assert response.status_code == 200
    assert response.json() == {
        "date": {
            "1": "2021-12-12",
            "0": "2022-12-12",
        },
        "title": {
            "1": "test2",
            "0": "test3",
        },
        "cate": {
            "1": "test2",
            "0": "test3",
        },
        "region": {
            "1": "test2",
            "0": "test3",
        },
        "wantWatchPeopleNum": {
            "1": "2人",
            "0": "3人",
        },
    }


def test_get_playing():
    response = client.get("/now_playing_out_filter", params={"args": "全部, 3, False"})
    assert response.status_code == 200
    assert response.json() == {
        "title": {
            "0": "test1",
            "1": "test2",
            "2": "test3",
        },
        "score": {
            "0": "1.1",
            "1": "2.2",
            "2": "3.3",
        },
        "region": {
            "0": "test1",
            "1": "test2",
            "2": "test3",
        },
        "director": {
            "0": "test1",
            "1": "test2",
            "2": "test3",
        },
        "actors": {
            "0": "test1",
            "1": "test2",
            "2": "test3",
        },
    }


def test_detail():
    response = client.get("/print_detail", params={"args": "test1"})
    assert response.status_code == 200
    assert response.json() == "test1test1"
