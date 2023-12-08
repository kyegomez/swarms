import requests
from lxml import etree
import pandas as pd
import re
from ...tool import Tool
from typing import List
from typing_extensions import TypedDict


class ComingMovieInfo(TypedDict):
    date: str
    title: str
    cate: str
    region: str
    wantWatchPeopleNum: str
    link: str


class PlayingMovieInfo(TypedDict):
    title: str
    score: str
    region: str
    director: str
    actors: str
    link: str


class DoubanAPI:
    def __init__(self) -> None:
        self._endpoint = "https://movie.douban.com"
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/108.0.0.0 Safari/537.36"
        }

    def fetch_page(self, url: str):
        """fetch_page(url: str) print html text of url"""
        s = requests.session()
        s.keep_alive = False
        response = s.get(url, headers=self._headers, verify=False)

        return response

    def get_coming(self) -> List[ComingMovieInfo]:
        response = self.fetch_page(f"{self._endpoint}/coming")
        ret: List[ComingMovieInfo] = []

        parser = etree.HTMLParser(encoding="utf-8")
        tree = etree.HTML(response.text, parser=parser)

        movies_table_path = '//*[@id="content"]/div/div[1]/table/tbody'
        movies_table = tree.xpath(movies_table_path)
        for filmChild in movies_table[0].iter("tr"):
            filmTime = filmChild.xpath("td[1]/text()")[0].strip()
            filmName = filmChild.xpath("td[2]/a/text()")[0]
            filmType = filmChild.xpath("td[3]/text()")[0].strip()
            filmRegion = filmChild.xpath("td[4]/text()")[0].strip()
            filmWantWatching = filmChild.xpath("td[5]/text()")[0].strip()
            filmLink = filmChild.xpath("td[2]/a/@href")[0]
            ret.append(
                ComingMovieInfo(
                    date=filmTime,
                    title=filmName,
                    cate=filmType,
                    region=filmRegion,
                    wantWatchPeopleNum=filmWantWatching,
                    link=filmLink,
                )
            )
        return ret

    def get_now_playing(self) -> List[PlayingMovieInfo]:
        # Get the movie list currently on show, the movie list of different cities is the same
        response = self.fetch_page(f"{self._endpoint}/cinema/nowplaying/beijing/")
        ret: List[PlayingMovieInfo] = []

        parser = etree.HTMLParser(encoding="utf-8")
        tree = etree.HTML(response.text, parser=parser)

        movies_table_path = './/div[@id="nowplaying"]/div[2]/ul'
        movies_table = tree.xpath(movies_table_path)
        for filmChild in movies_table[0]:
            filmName = filmChild.xpath("@data-title")[0]
            filmScore = filmChild.xpath("@data-score")[0]
            filmRegion = filmChild.xpath("@data-region")[0]
            filmDirector = filmChild.xpath("@data-director")[0]
            filmActors = filmChild.xpath("@data-actors")[0]
            filmLink = filmChild.xpath("ul/li[1]/a/@href")[0]
            ret.append(
                PlayingMovieInfo(
                    title=filmName,
                    score=filmScore,
                    region=filmRegion,
                    director=filmDirector,
                    actors=filmActors,
                    link=filmLink,
                )
            )
        return ret

    def get_movie_detail(self, url: str) -> str:
        response = self.fetch_page(url)
        parser = etree.HTMLParser(encoding="utf-8")
        tree = etree.HTML(response.text, parser=parser)
        info_path = './/div[@class="subject clearfix"]/div[2]'

        director = tree.xpath(f"{info_path}/span[1]/span[2]/a/text()")[0]

        actors = []
        actors_spans = tree.xpath(f"{info_path}/span[3]/span[2]")[0]
        for actors_span in actors_spans:
            actors.append(actors_span.text)
        actors = "、".join(actors[:3])

        types = []
        spans = tree.xpath(f"{info_path}")[0]
        for span in spans.iter("span"):
            if "property" in span.attrib and span.attrib["property"] == "v:genre":
                types.append(span.text)
        types = "、".join(types)

        for span in spans:
            if span.text == "制片国家/地区:":
                region = span.tail.strip()
                break
        Synopsis = tree.xpath('.//div[@class="related-info"]/div/span')[0].text.strip()
        detail = f"是一部{region}的{types}电影，由{director}导演，{actors}等人主演.\n剧情简介：{Synopsis}"
        return detail


def build_tool(config) -> Tool:
    tool = Tool(
        "Film Search Plugin",
        "search for up-to-date film information.",
        name_for_model="Film Search",
        description_for_model="Plugin for search for up-to-date film information.",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    if "debug" in config and config["debug"]:
        douban_api = config["douban_api"]
    else:
        douban_api = DoubanAPI()

    @tool.get("/coming_out_filter")
    def coming_out_filter(args: str):
        """coming_out_filter(args: str) prints the details of the filtered [outNum] coming films now according to region, cate and outNum.
        args is a list like 'str1, str2, str3, str4'
        str1 represents Production country or region. If you cannot find a region, str1 is 全部
        str2 represents movie's category. If you cannot find a category, str2 is 全部
        str3 can be a integer number that agent want to get. If you cannot find a number, str2 is 100. If the found movie's num is less than str2, Final Answer only print [the found movie's num] movies.
        str4 can be a True or False that refluct whether agent want the result sorted by people number which look forward to the movie.
        Final answer should be complete.

        This is an example:
        Thought: I need to find the upcoming Chinese drama movies and the top 2 most wanted movies
        Action: coming_out_filter
        Action Input: {"args" : "中国, 剧情, 2, True"}
        Observation: {"date":{"23":"04月28日","50":"07月"},"title":{"23":"长空之王","50":"热烈"},"cate":{"23":"剧情 / 动作","50":"剧情 / 喜剧"},"region":{"23":"中国大陆","50":"中国大陆"},"wantWatchPeopleNum":{"23":"39303人","50":"26831人"}}
        Thought: I now know the top 2 upcoming Chinese drama movies
        Final Answer: 即将上映的中国剧情电影有2部：长空之王、热烈，大家最想看的前2部分别是：长空之王、热烈。
        """
        args = re.findall(r"\b\w+\b", args)
        region = args[0]
        if region == "全部":
            region = ""
        cate = args[1]
        if cate == "全部":
            cate = ""
        outNum = int(args[2])
        WantSort = True if args[3] == "True" else False

        coming_movies = []
        for movie in douban_api.get_coming():
            if (cate in movie["cate"]) and (region in movie["region"]):
                coming_movies.append(
                    {
                        "date": movie["date"],
                        "title": movie["title"],
                        "cate": movie["cate"],
                        "region": movie["region"],
                        "wantWatchPeopleNum": int(
                            movie["wantWatchPeopleNum"].replace("人", "")
                        ),
                        "link": movie["link"],
                    }
                )

        # Sort by people that are looking forward to the movie
        if WantSort:
            coming_movies = sorted(
                coming_movies, key=lambda x: x["wantWatchPeopleNum"], reverse=True
            )

        ret = {
            "date": {},
            "title": {},
            "cate": {},
            "region": {},
            "wantWatchPeopleNum": {},
        }
        for i, movie in enumerate(coming_movies[:outNum]):
            i = str(i)
            ret["date"][i] = movie["date"]
            ret["title"][i] = movie["title"]
            ret["cate"][i] = movie["cate"]
            ret["region"][i] = movie["region"]
            ret["wantWatchPeopleNum"][i] = "{}人".format(movie["wantWatchPeopleNum"])
        return ret

    @tool.get("/now_playing_out_filter")
    def now_playing_out_filter(args: str):
        """NowPlayingOutFilter(args: str) prints the details of the filtered [outNum] playing films now according to region, scoreSort
        args is a list like 'str1, str2, str3'
        str1 can be '中国','日本' or other Production country or region. If you cannot find a region, str1 is 全部
        str2 can be a integer number that agent want to get. If you cannot find a number, str2 is 100. If the found movie's num is less than str2, Final Answer only print [the found movie's num] movies.
        str3 can be a True or False that refluct whether agent want the result sorted by score.
        Final answer should be complete.

        This is an example:
        Input: 您知道现在有正在上映中国的电影吗？请输出3部
        Thought: I need to find the currently playing movies with the highest scores
        Action: now_playing_out_filter
        Action Input: {"args" : "全部, 3, True"}
        Observation: {"title":{"34":"切腹","53":"吉赛尔","31":"小森林 夏秋篇"},"score":{"34":"9.4","53":"9.2","31":"9.0"},"region":{"34":"日本","53":"西德","31":"日本"},"director":{"34":"小林正树","53":"Hugo Niebeling","31":"森淳一"},"actors":{"34":"仲代达矢 / 石浜朗 / 岩下志麻","53":"卡拉·弗拉奇 / 埃里克·布鲁恩 / Bruce Marks","31":"桥本爱 / 三浦贵大 / 松冈茉优"}}
        Thought: I now know the currently playing movies with the highest scores
        Final Answer: 现在上映的评分最高的3部电影是：切腹、吉赛尔、小森林 夏秋篇

        """
        args = re.findall(r"\b\w+\b", args)
        region = args[0]
        if region == "全部":
            region = ""
        outNum = int(args[1])
        scoreSort = True if args[2] == "True" else False

        playing_movies = []
        for movie in douban_api.get_now_playing():
            if region in movie["region"]:
                playing_movies.append(
                    {
                        "title": movie["title"],
                        "score": float(movie["score"]),
                        "region": movie["region"],
                        "director": movie["director"],
                        "actors": movie["actors"],
                        "link": movie["link"],
                    }
                )

        # Sort by score
        if scoreSort:
            playing_movies = sorted(
                playing_movies, key=lambda x: x["score"], reverse=True
            )

        ret = {
            "title": {},
            "score": {},
            "region": {},
            "director": {},
            "actors": {},
        }
        for i, movie in enumerate(playing_movies[:outNum]):
            i = str(i)
            ret["title"][i] = movie["title"]
            ret["score"][i] = "{}".format(movie["score"])
            ret["region"][i] = movie["region"]
            ret["director"][i] = movie["director"]
            ret["actors"][i] = movie["actors"]
        return ret

    @tool.get("/print_detail")
    def print_detail(args: str):
        """parsing_detail_page(args) prints the details of a movie, giving its name.
        args is a list like 'str1'
        str1 is target movie's name.
        step1: apply function parse_coming_page and parse_nowplaying_page and get all movie's links and other infomation.
        step2: get the target movie's link from df_coming or df_nowplaying
        step3: get detail from step2's link

        This is an example:
        Input: "电影流浪地球2怎么样？"
        Thought: I need to find the movie's information
        Action: print_detail
        Action Input: {"args" : "流浪地球2"}
        Observation: "是一部中国大陆的科幻、冒险、灾难电影，由郭帆导演，吴京、刘德华、李雪健等人主演.\n剧情简介：太阳即将毁灭，人类在地球表面建造出巨大的推进器，寻找新的家园。然而宇宙之路危机四伏，为了拯救地球，流浪地球时代的年轻人再次挺身而出，展开争分夺秒的生死之战。"
        Thought: I now know the final answer
        Final Answer: 流浪地球2是一部中国大陆的科幻、冒险、灾难电影，由郭帆导演，吴京、刘德华、李雪健等人主演，剧情简介是太阳即将毁灭，人类在地球表面建造出巨大的推进器，寻找新的家园，然而宇宙之路危机四伏，为了拯救地球，流浪地球时代的年轻人再次挺身而出，

        """
        args = re.findall(r"\b\w+\b", args)
        filmName = args[0]

        link = None

        if link is None:
            for movie in douban_api.get_coming():
                if movie["title"] == filmName:
                    link = movie["link"]
                    break

        if link is None:
            for movie in douban_api.get_now_playing():
                if movie["title"] == filmName:
                    link = movie["link"]
                    break

        if link is None:
            return "没有找到该电影"

        return "{}{}".format(filmName, douban_api.get_movie_detail(link))

    return tool
