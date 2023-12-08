import requests
from bs4 import BeautifulSoup
from ..tool import Tool
from fastapi import Request
from uuid import UUID


def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


class WikiPage:
    def __init__(self):
        self.page = ""
        self.paragraphs = []
        self.sentences = []
        self.lookup_cnt = 0
        self.lookup_list = []
        self.lookup_keyword = None

    def reset_page(self):
        self.page = ""
        self.paragraphs = []
        self.sentences = []
        self.lookup_cnt = 0
        self.lookup_list = []
        self.lookup_keyword = None

    def get_page_obs(self, page):
        self.page = page
        paragraphs = []
        sentences = []
        # find all paragraphs
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        self.paragraphs = paragraphs
        self.sentences = sentences
        return " ".join(sentences[:5])

    def construct_lookup_list(self, keyword: str):
        sentences = self.sentences
        parts = []
        for index, p in enumerate(sentences):
            if keyword.lower() in p.lower():
                parts.append(index)
        self.lookup_list = parts
        self.lookup_keyword = keyword
        self.lookup_cnt = 0


currentPage = WikiPage()


def build_tool(config) -> Tool:
    tool = Tool(
        "Wikipedia Search",
        "Plugin for wikipedia",
        name_for_model="Wikipedia",
        name_for_human="Wikipedia",
        description_for_model="A tool to search entity, view content and disambiguate entity on Wikipedia.\nCurrent endpoint for each function is simple and you should only use exact entity name as input for search and disambiguate. And the keyword input to lookup api should also be simple like one or two words.\nSome Tips to use the APIs bertter:\n1. When the search api doesn't find the corresponding page, you should search a related entity in the return list.\n2. You can only search one entity name in each action, so, don't concat multiple entity names in one search input.\n3. The lookup api can only be used after search api since it depends on the result page of search.\n4. When search api result in an entity page that is not related, you should disambiguate the searched entity to find other entities with the same name.\n5. Don't over rely one this simple tool, you may figure out the next action based on your own knowledge.",
        description_for_human="A tool to search entity, view content and disambiguate entity on Wikipedia.\nCurrent endpoint for each function is simple and you should only use exact entity name as input for search and disambiguate. And the keyword input to lookup api should also be simple like one or two words.",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    @tool.get("/search")
    def search(entity: str, request: Request):
        """The input is an exact entity name. The action will search this entity name on Wikipedia and returns the first five sentences if it exists. If not, it will return some related entities to search next."""

        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        response_text = requests.get(search_url).text
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:  # mismatch
            result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            obs = f"Could not find {entity}. Similar: {result_titles[:5]}."
        else:
            local_page = [
                p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")
            ]
            if any("may refer to:" in p for p in local_page):
                obs = search("[" + entity + "]", request)
            else:
                currentPage.reset_page()
                page = ""
                for p in local_page:
                    if len(p.split(" ")) > 2:
                        page += clean_str(p)
                    if not p.endswith("\n"):
                        page += "\n"
                obs = currentPage.get_page_obs(page)
        return obs

    @tool.get("/lookup")
    def lookup(keyword: str, request: Request) -> str:
        """The input is a keyword. This action will look up in the current passage and return the next several sentences containing the keyword in current passage."""
        # lookup_keyword = request.session["lookup_keyword"]
        # lookup_list = request.session["lookup_list"]
        # lookup_cnt = request.session["lookup_cnt"]
        # sentences = request.session["sentences"]
        lookup_keyword = currentPage.lookup_keyword
        if lookup_keyword != keyword:  # reset lookup
            currentPage.construct_lookup_list(keyword)
        lookup_list = currentPage.lookup_list
        lookup_cnt = currentPage.lookup_cnt
        sentences = currentPage.sentences
        if lookup_cnt >= len(lookup_list):
            obs = "No more results."
        else:
            index = lookup_list[lookup_cnt]
            before_sentence_num = min(index, 1)
            max_sentence_num = 3  # 一共3句话
            lookup_result = " ".join(
                sentences[
                    index
                    - before_sentence_num : index
                    - before_sentence_num
                    + max_sentence_num
                ]
            )
            obs = f"(Result {lookup_cnt + 1} / {len(lookup_list)}) " + lookup_result
            currentPage.lookup_cnt += 1
        return obs

    @tool.get("/disambiguation")
    def disambiguation(entity: str) -> str:
        """The input is an entity name. This action will disambiguate this entity name to find other entities with similar names in Wikipedia."""
        # self.reset_pages()
        url = f"https://en.wikipedia.org/wiki/{entity}_(disambiguation)"
        # url = f"https://en.wikipedia.org{href}"
        response = requests.get(url)
        html_code = response.content
        soup = BeautifulSoup(html_code, "html.parser")
        # Extract all the list items from the page
        list_items = soup.find_all("li")
        # Extract the text content of each list item and print it
        titles = []
        for item in list_items:
            link = item.find("a")
            if (
                link
                and entity.lower() in item.get_text().lower()
                and "/wiki" in link["href"]
            ):
                titles.append(link.get_text())
                # print(f"{link.get_text()} - {link['href']}")
                # print(item.get_text())
                # print("\n")
                whether_need_disambiguation = True
        max_return_titles = 5
        if len(titles) > max_return_titles:
            titles = titles[:max_return_titles]
        obs = f"Related entities to {entity}: {titles}"
        return obs

    return tool
