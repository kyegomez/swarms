import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import html2text
from concurrent.futures import ThreadPoolExecutor, as_completed
from playwright.sync_api import sync_playwright
import time
from tenacity import retry, stop_after_attempt, wait_exponential

console = Console()
load_dotenv()

class WebsiteChecker:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cx = os.getenv("GOOGLE_CX")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.outputs_dir = "outputs"
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        # Initialize html2text
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = True
        
        # Configure retry settings
        self.max_retries = 3
        self.max_threads = 10  # Concurrent threads
        self.timeout = 15  # Seconds

    async def fetch_search_results(self, query: str) -> List[Dict]:
        """Fetch top 10 search results using Google Custom Search API"""
        async with aiohttp.ClientSession() as session:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cx,
                "q": query,
                "num": 10  # Fetch top 10 results
            }
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        for item in data.get("items", []):
                            if "link" in item and not any(x in item["link"].lower() for x in [".pdf", ".doc", ".docx"]):
                                results.append({
                                    "title": item.get("title", ""),
                                    "link": item["link"],
                                    "snippet": item.get("snippet", "")
                                })
                        return results[:10]  # Ensure we only take top 10
                    else:
                        console.print(f"[red]Error: {response.status} - {await response.text()}[/red]")
                        return []
            except Exception as e:
                console.print(f"[red]Error fetching search results: {str(e)}[/red]")
                return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def extract_content_with_retry(self, url: str) -> Optional[Dict]:
        """Extract content from a URL with retry mechanism"""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                
                page = context.new_page()
                page.set_default_timeout(25000)  # 10 second timeout
                
                page.goto(url)
                page.wait_for_load_state('networkidle', timeout=20000)
                
                # Extract content
                content = page.content()
                soup = BeautifulSoup(content, 'lxml')
                
                # Clean up content
                for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    element.decompose()
                
                # Get main content
                main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': ['content', 'main']})
                if not main_content:
                    main_content = soup.find('body')
                
                # Convert to markdown-like text
                clean_text = self.html_converter.handle(str(main_content))
                
                browser.close()
                
                return {
                    "url": url,
                    "title": soup.title.string if soup.title else "No title",
                    "content": clean_text.strip()
                }
                
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to extract from {url}: {str(e)}[/yellow]")
            return None

    def process_url(self, url: str) -> Optional[Dict]:
        """Process a single URL with progress tracking"""
        try:
            return self.extract_content_with_retry(url)
        except Exception as e:
            console.print(f"[red]Failed to process {url}: {str(e)}[/red]")
            return None

    async def process_urls_concurrent(self, urls: List[str]) -> List[Dict]:
        """Process multiple URLs concurrently using ThreadPoolExecutor"""
        successful_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Processing websites...", total=len(urls))
            
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                future_to_url = {executor.submit(self.process_url, url): url for url in urls}
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        if result:
                            successful_results.append(result)
                    except Exception as e:
                        console.print(f"[red]Error processing {url}: {str(e)}[/red]")
                    finally:
                        progress.advance(task)
        
        return successful_results

    async def summarize_with_gemini(self, extracted_data: List[Dict], query: str) -> str:
        """Generate summary using Gemini API"""
        genai.configure(api_key=self.gemini_api_key)
        
        # Format content for summarization
        formatted_content = "# Source Materials:\n\n"
        for i, item in enumerate(extracted_data, 1):
            formatted_content += f"""
### Source {i}: {item['title']}
URL: {item['url']}

{item['content'][:2000]}  # Limit content length per source

---
"""
        
        prompt = f"""
Analyze and summarize the following content about: "{query}"

Create a detailed summary with these sections:
1. Key Findings (2-3 paragraphs)
2. Important Details (bullet points)
3. Sources (numbered list)

Focus on accuracy, clarity, and completeness.
Present conflicting information if found.
Use proper markdown formatting.

Content to analyze:
{formatted_content}
"""

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
        )

        response = await asyncio.to_thread(
            lambda: model.generate_content(prompt).text
        )
        
        return response

    async def search(self, query: str) -> str:
        """Main search function with timing"""
        start_time = time.time()
        
        console.print(f"\n[bold cyan]Searching for: {query}[/bold cyan]\n")
        
        # Fetch search results
        search_results = await self.fetch_search_results(query)
        if not search_results:
            return "No search results found."
        
        # Extract URLs
        urls = [result["link"] for result in search_results]
        
        # Process URLs concurrently
        extracted_data = await self.process_urls_concurrent(urls)
        
        # Generate summary
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("[cyan]Generating summary...", total=None)
            summary = await self.summarize_with_gemini(extracted_data, query)
            progress.update(task, completed=True)
        
        # Save results
        results = {
            "query": query,
            "search_results": search_results,
            "extracted_data": extracted_data,
            "summary": summary
        }
        
        with open(os.path.join(self.outputs_dir, "search_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print results
        console.print("\n[bold green]====== Search Summary ======[/bold green]\n")
        console.print(summary)
        console.print("\n[bold green]========================[/bold green]")
        console.print(f"\n[bold cyan]Execution time: {execution_time:.2f} seconds[/bold cyan]\n")
        
        return summary

def search(query: str) -> str:
    """Synchronous wrapper for the async search function"""
    checker = WebsiteChecker()
    return asyncio.run(checker.search(query))

# search_tool_schema = functions_to_openai_tools([search])
# # tools = functions_to_openai_tools([search, get_weather])

# # Print the generated schemas
# print(json.dumps(tools, indent=2))
# if __name__ == "__main__":
#     query = input("Enter your search query: ")
#     result = search(query)

# search("who won elections 2024 us")