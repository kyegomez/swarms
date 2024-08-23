import re
from urllib.parse import urlparse, urljoin
import redis
from firecrawl import FirecrawlApp
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery

class RedisVectorStorage:
    """  Provides vector storage database operations using Redis """
    def __init__(self, context: str="swarms", use_gpu=False):
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
            "prefix": "chunk"
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
            }
        ]
        })

        self.schema = schema
        self.index = SearchIndex(self.schema, self.redis_client)
        self.index.create()

    # Function to extract Markdown links
    def extract_markdown_links(self, markdown_text):
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = re.findall(pattern, markdown_text)
        urls = [link[1] for link in links]
        return urls

    # Function to check if a URL is internal to the initial domain
    def is_internal_link(self, url, base_domain):
        parsed_url = urlparse(url)
        return parsed_url.netloc == '' or parsed_url.netloc == base_domain

    # Function to split markdown content into chunks of max 5000 characters at natural breakpoints
    def split_markdown_content(self, markdown_text, max_length=5000):
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

    # Function to store chunks and their embeddings in Redis
    def store_chunks_in_redis(self, url, chunks):
        data = []
        for i, chunk in enumerate(chunks):
            embedding = self.vectorizer.embed(chunk, input_type="search_document", as_buffer=True)
            data.append({
                "id": f"{url}::chunk::{i+1}",
                "content": chunk,
                "content_embedding": embedding
            })
        self.index.load(data)
        print(f"Stored {len(chunks)} chunks for URL {url} in Redis.")

    # Function to recursively crawl a URL and its Markdown links
    def crawl_recursive(self, url, base_domain, visited=None):
        if visited is None:
            visited = set()

        if url in visited:
            return
        visited.add(url)

        # Check if the URL has already been processed
        if self.redis_client.exists(f"{url}::chunk::1"):
            print(f"URL {url} has already been processed. Skipping.")
            return

        print(f"Crawling URL: {url}")

        params = {
            'pageOptions': {
                'onlyMainContent': False,
                'fetchPageContent': True,
                'includeHTML': True,
            }
        }
        crawl_result = self.app.crawl_url(url, params=params, wait_until_done=True)

        for result in crawl_result:
            print("Content:\n\n")
            markdown_content = result["markdown"]

            # Split markdown content into natural chunks
            chunks = self.split_markdown_content(markdown_content)

            # Store the chunks and their embeddings in Redis
            self.store_chunks_in_redis(url, chunks)

            links = self.extract_markdown_links(markdown_content)
            print("Extracted Links:", links)

            for link in links:
                absolute_link = urljoin(url, link)
                if self.is_internal_link(absolute_link, base_domain):
                    self.crawl_recursive(absolute_link, base_domain, visited)

    # Function to embed a string and perform a Redis vector database query
    def embed(self, query: str, num_results: int=3):
        query_embedding = self.vectorizer.embed(query)

        vector_query = VectorQuery(
            vector=query_embedding,
            vector_field_name="content_embedding",
            num_results=num_results,
            return_fields=["id", "content"],
            return_score=True
        )

        # show the raw redis query
        results = self.index.query(vector_query)
        return results

    def crawl(self, crawl_url: str):
        # Start the recursive crawling from the initial URL
        base_domain = urlparse(crawl_url).netloc
        self.crawl_recursive(crawl_url, base_domain)

if __name__ == "__main__":
    storage = RedisVectorStorage()
    storage.crawl("https://docs.swarms.world/en/latest/")
