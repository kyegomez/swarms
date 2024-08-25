""" Vector storage using Firecrawl and Redis """
import re
from urllib.parse import urlparse, urljoin
import redis
from firecrawl import FirecrawlApp
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query.filter import Tag
from redisvl.query import VectorQuery, FilterQuery

class RedisVectorStorage:
    """  Provides vector storage database operations using Redis """
    def __init__(self, context: str="swarms", use_gpu=False, overwrite=False):
        self.use_gpu = use_gpu
        self.context = context
        # Initialize the FirecrawlApp with your API key
        self.app = FirecrawlApp(
            api_key="EMPTY",
            api_url="http://localhost:3002")  # EMPTY for localhost

        # Connect to the local Redis server
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

        # Initialize the Cohere text vectorizer
        self.vectorizer = HFTextVectorizer()

        index_name = self.context

        schema = IndexSchema.from_dict({
        "index": {
            "name": index_name,
        },
        "fields": [
            {
                "name": "id",
                "type": "tag",
                "attrs": {
                    "sortable": True
                }
            },
            {
                "name": "content",
                "type": "text",
                "attrs": {
                    "sortable": True
                }
            },
            {
                "name": "content_embedding",
                "type": "vector",
                "attrs": {
                    "dims": self.vectorizer.dims,
                    "distance_metric": "cosine",
                    "algorithm": "hnsw",
                    "datatype": "float32"
                }
            },
            {
                "name": "source_url",
                "type": "text",
                "attrs": {
                    "sortable": True
                }
            }
        ]
        })

        self.schema = schema
        self.index = SearchIndex(self.schema, self.redis_client)
        self.index.create(overwrite=overwrite, drop=overwrite)

    def extract_markdown_links(self, markdown_text):
        """ Extract Markdown links from the given markdown text """
        pattern = r'\[([^\]]+)\]\(([^)]+?)(?:\s+"[^"]*")?\)'
        links = re.findall(pattern, markdown_text)
        urls = [link[1] for link in links]
        return urls

    def is_internal_link(self, url: str, base_domain: str):
        """ Check if a URL is internal to the initial domain """
        if (url == '\\' or url.startswith("mailto")):
            return False
        parsed_url = urlparse(url)
        return parsed_url.netloc == '' or parsed_url.netloc == base_domain

    def split_markdown_content(self, markdown_text, max_length=5000):
        """ Split markdown content into chunks of max 5000 characters at natural breakpoints """
        paragraphs = markdown_text.split('\n\n')  # Split by paragraphs
        chunks = []
        current_chunk = ''

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_length:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                current_chunk += '\n\n' + paragraph

            if len(paragraph) > max_length:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    if len(sentence) > max_length:
                        chunks.append(sentence[:max_length])
                        current_chunk = sentence[max_length:]
                    elif len(current_chunk) + len(sentence) > max_length:
                        chunks.append(current_chunk)
                        current_chunk = sentence
                    else:
                        current_chunk += ' ' + sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def store_chunks_in_redis(self, url, chunks):
        """ Store chunks and their embeddings in Redis """
        parsed_url = urlparse(url)
        trimmed_url = parsed_url.netloc + parsed_url.path  # Remove scheme (http:// or https://)

        data = []
        for i, chunk in enumerate(chunks):
            embedding = self.vectorizer.embed(chunk, input_type="search_document", as_buffer=True)
            data.append({
                "id": f"{trimmed_url}::chunk::{i+1}",
                "content": chunk,
                "content_embedding": embedding,
                "source_url": trimmed_url
            })
        self.index.load(data)
        print(f"Stored {len(chunks)} chunks for URL {url} in Redis.")

    def crawl_iterative(self, start_url, base_domain):
        """ Iteratively crawl a URL and its Markdown links """
        visited = set()
        stack = [start_url]

        while stack:
            url = stack.pop()
            if url in visited:
                continue

            parsed_url = urlparse(url)
            trimmed_url = parsed_url.netloc + parsed_url.path  # Remove scheme (http:// or https://)

            # Check if the URL has already been processed
            t = Tag("id") == f"{trimmed_url}::chunk::1"  # Use the original URL format

            # Use a simple filter query instead of a vector query
            filter_query = FilterQuery(filter_expression=t)
            results = self.index.query(filter_query)
            if results:
                print(f"URL {url} has already been processed. Skipping.")
                visited.add(url)
                continue

            print(f"Crawling URL: {url}")

            params = {
                'pageOptions': {
                    'onlyMainContent': False,
                    'fetchPageContent': True,
                    'includeHTML': False,
                }
            }

            crawl_result = []
            if self.is_internal_link(url, base_domain) and not url in visited:
                crawl_result.append(self.app.scrape_url(url, params=params))
                visited.add(url)

                for result in crawl_result:
                    markdown_content = result["markdown"]
                    result_url = result["metadata"]["sourceURL"]
                    print("Markdown sourceURL: " + result_url)
                    # print("Content:\n\n")
                    # print(markdown_content)
                    # print("\n\n")
                    # Split markdown content into natural chunks
                    chunks = self.split_markdown_content(markdown_content)

                    # Store the chunks and their embeddings in Redis
                    self.store_chunks_in_redis(result_url, chunks)

                    links = self.extract_markdown_links(markdown_content)
                    print("Extracted Links:", links)
                    # print("Extracted Links:", links)

                    for link in links:
                        absolute_link = urljoin(result_url, link)
                        if self.is_internal_link(absolute_link, base_domain):
                            if absolute_link not in visited:
                                stack.append(absolute_link)
                                print("Appended link: " + absolute_link)
                        else:
                            visited.add(absolute_link)

    def crawl(self, crawl_url: str):
        """ Start the iterative crawling from the initial URL """
        base_domain = urlparse(crawl_url).netloc
        self.crawl_iterative(crawl_url, base_domain)

    def embed(self, query: str, num_results: int=3):
        """ Embed a string and perform a Redis vector database query """
        query_embedding = self.vectorizer.embed(query)

        vector_query = VectorQuery(
            vector=query_embedding,
            vector_field_name="content_embedding",
            num_results=num_results,
            return_fields=["id", "content", "source_url"],
            return_score=True
        )

        # show the raw redis query
        results = self.index.query(vector_query)
        return results

if __name__ == "__main__":
    storage = RedisVectorStorage(overwrite=False)
    storage.crawl("https://docs.swarms.world/en/latest/")
    responses = storage.embed("What is Swarms, and how do I install swarms?", 5)
    for response in responses:
        encoded_id = response['id']  # Access the 'id' field directly
        source_url = response['source_url']
        print(f"Decoded ID: {encoded_id}, Source URL: {source_url}")
