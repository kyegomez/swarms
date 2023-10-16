"""
Paper Swarm
1. Scrape https://huggingface.co/papers for all papers, by search for all links on the paper with a /papers/, then clicks, gets the header, and then the abstract.
and various links and then adds them to a txt file for each paper on https://huggingface.co/papers

2. Feed prompts iteratively into Anthropic for summarizations + value score on impact, reliability, and novel, and other paper ranking mechanisms

3. Store papers in a database with metadata. Agents can use retrieval

4. Discord Bot // Twitter Bot
"""


import requests
from bs4 import BeautifulSoup
import os


class Paper:
    def __init__(self, title, date, authors, abstract):
        self.title = title
        self.date = date
        self.authors = authors
        self.abstract = abstract


class Scraper:
    def __init__(self, url):
        self.url = url

    def get_paper_links(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, "html.parser")
        links = [
            a["href"] for a in soup.find_all("a", href=True) if "/papers/" in a["href"]
        ]
        return links

    def get_paper_details(self, link):
        response = requests.get(self.url + link)
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find("h1").text
        date_tag = soup.find("time")
        date = date_tag.text if date_tag else "Unknown"
        authors = [author.text for author in soup.find_all("span", class_="author")]
        abstract_tag = soup.find("div", class_="abstract")
        abstract = abstract_tag.text if abstract_tag else "Abstract not found"
        return Paper(title, date, authors, abstract)


class FileWriter:
    def __init__(self, directory):
        self.directory = directory

    def write_paper(self, paper):
        with open(os.path.join(self.directory, paper.title + ".txt"), "w") as f:
            f.write(f"h1: {paper.title}\n")
            f.write(f"Published on {paper.date}\n")
            f.write("Authors:\n")
            for author in paper.authors:
                f.write(f"{author}\n")
            f.write("Abstract\n")
            f.write(paper.abstract)


scraper = Scraper("https://huggingface.co/papers")
file_writer = FileWriter("images")

links = scraper.get_paper_links()
for link in links:
    paper = scraper.get_paper_details(link)
    file_writer.write_paper(paper)
