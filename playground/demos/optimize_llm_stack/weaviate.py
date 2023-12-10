from swarms.memory import WeaviateClient

weaviate_client = WeaviateClient(
    http_host="YOUR_HTTP_HOST",
    http_port="YOUR_HTTP_PORT",
    http_secure=True,
    grpc_host="YOUR_gRPC_HOST",
    grpc_port="YOUR_gRPC_PORT",
    grpc_secure=True,
    auth_client_secret="YOUR_APIKEY",
    additional_headers={"X-OpenAI-Api-Key": "YOUR_OPENAI_APIKEY"},
    additional_config=None,  # You can pass additional configuration here
)

