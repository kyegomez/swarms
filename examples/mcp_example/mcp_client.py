        from swarms import Agent
        from swarms.tools.mcp_integration import MCPServerSseParams
        from loguru import logger

        # Comprehensive math prompt that encourages proper JSON formatting
        MATH_AGENT_PROMPT = """
        You are a helpful math calculator assistant.

        Your role is to understand natural language math requests and perform calculations.
        When asked to perform calculations:

        1. Determine the operation (add, multiply, or divide)
        2. Extract the numbers from the request
        3. Use the appropriate math operation tool

        FORMAT YOUR TOOL CALLS AS JSON with this format:
        {"tool_name": "add", "a": <first_number>, "b": <second_number>}
        or
        {"tool_name": "multiply", "a": <first_number>, "b": <second_number>}
        or
        {"tool_name": "divide", "a": <first_number>, "b": <second_number>}

        Always respond with a tool call in JSON format first, followed by a brief explanation.
        """

        def initialize_math_system():
            """Initialize the math agent with MCP server configuration."""
            # Configure the MCP server connection
            math_server = MCPServerSseParams(
                url="http://0.0.0.0:8000",
                headers={"Content-Type": "application/json"},
                timeout=5.0,
                sse_read_timeout=30.0
            )

            # Create the agent with the MCP server configuration
            math_agent = Agent(
                agent_name="Math Assistant",
                agent_description="Friendly math calculator",
                system_prompt=MATH_AGENT_PROMPT,
                max_loops=1,
                mcp_servers=[math_server],  # Pass MCP server config as a list
                model_name="gpt-3.5-turbo",
                verbose=True  # Enable verbose mode to see more details
            )

            return math_agent

        def main():
            try:
                logger.info("Initializing math system...")
                math_agent = initialize_math_system()

                print("\nMath Calculator Ready!")
                print("Ask me any math question!")
                print("Examples: 'what is 5 plus 3?' or 'can you multiply 4 and 6?'")
                print("Type 'exit' to quit\n")

                while True:
                    try:
                        query = input("What would you like to calculate? ").strip()
                        if not query:
                            continue
                        if query.lower() == 'exit':
                            break

                        logger.info(f"Processing query: {query}")
                        result = math_agent.run(query)
                        print(f"\nResult: {result}\n")

                    except KeyboardInterrupt:
                        print("\nGoodbye!")
                        break
                    except Exception as e:
                        logger.error(f"Error processing query: {e}")
                        print(f"Sorry, there was an error: {str(e)}")

            except Exception as e:
                logger.error(f"System initialization error: {e}")
                print(f"Failed to start the math system: {str(e)}")

        if __name__ == "__main__":
            main()