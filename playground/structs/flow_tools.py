from swarms.models import Anthropic
from swarms.structs import Flow
from swarms.tools.tool import tool

import asyncio


llm = Anthropic(
    anthropic_api_key="",
)


async def async_load_playwright(url: str) -> str:
    """Load the specified URLs using Playwright and parse using BeautifulSoup."""
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright

    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (
                phrase.strip() for line in lines for phrase in line.split("  ")
            )
            results = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results


def run_async(coro):
    event_loop = asyncio.get_event_loop()
    return event_loop.run_until_complete(coro)


@tool
def browse_web_page(url: str) -> str:
    """Verbose way to scrape a whole webpage. Likely to cause issues parsing."""
    return run_async(async_load_playwright(url))


## Initialize the workflow
flow = Flow(
    llm=llm,
    max_loops=5,
    tools=[browse_web_page],
    dashboard=True,
)

out = flow.run(
    "Generate a 10,000 word blog on mental clarity and the benefits of"
    " meditation."
)
