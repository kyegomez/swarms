# Import the necessary libraries.
import asyncio
import websockets

# Create a list of public group chats.
public_group_chats = []


# Create a function to handle incoming websocket connections.
async def handle_websocket(websocket, path):
    # Get the username of the user.
    username = await websocket.recv()
    print(f"New connection from {username}")

    # Add the user to the list of public group chats.
    public_group_chats.append(websocket)

    try:
        # Wait for the user to send a message.
        while True:
            message = await websocket.recv()
            print(f"{username}: {message}")

            # Broadcast the message to all other users in the public group chats.
            for other_websocket in public_group_chats:
                if other_websocket != websocket:
                    await other_websocket.send(
                        f"{username}: {message}"
                    )
    finally:
        # Remove the user from the list of public group chats.
        public_group_chats.remove(websocket)
        print(f"{username} has disconnected")


# Create a websocket server.
server = websockets.serve(handle_websocket, "localhost", 8000)

# Run the websocket server.
asyncio.run(server)
