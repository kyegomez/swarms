
# Server-Bot

[View on Docker Hub](https://hub.docker.com/r/allenrkeen/server-bot)
### Discord bot to remotely monitor and control a docker based server. Using the docker socket.

Setup is pretty straightforward.
1. Create a new application in the *[discord developer portal](https://discord.com/developers/applications)*
2. Go to the bot section and click *Add Bot*
3. Reset Token and keep the token somewhere secure (This will be referred to as "DISCORD_TOKEN" in .env and docker environment variables)
4. Get the "Application ID" from the General Information tab of your application (This will be referred to as "DISCORD_CLIENT_ID" in .env and docker environment variables)
5. *Optional:* If you have developer mode enabled in Discord, get your server's ID by right-clicking on the server name and clicking "Copy ID" (This will be referred to as "DISCORD_GUILD_ID" in .env and docker environment variables)
   - If you skip this, it will still work, but commands will be published globally instead of to your server and can take up to an hour to be available in your server.
   - Using the Server ID will be more secure, making the commands available only in the specified server.
6. Run the application in your preffered method.
   - Run the docker container with the provided [docker-compose.yml](docker-compose.yml) or the docker run command below.

      ```bash
      docker run -v /var/run/docker.sock:/var/run/docker.sock --name server-bot \
      -e DISCORD_TOKEN=your_token_here \
      -e DISCORD_CLIENT_ID=your_client_id_here \
      -e DISCORD_GUILD_ID=your_guild_id_here \
      allenrkeen/server-bot:latest
      ```

   - Clone the repo, cd into the server-bot directory and run "npm install" to install dependencies, then "npm run start" to start the server
7. The program will build an invite link with the correct permissions and put it in the logs. Click the link and confirm the server to add the bot to.


Current commands:
  - /allcontainers
    - provides container name and status for all containers
  - /restartcontainer
    - provides an autocomplete list of running containers to select from, or just type in container name then restarts the container
  - /stopcontainer
    - provides an autocomplete list of running containers to select from, or just type in container name then stops the container
  - /startcontainer
    - provides an autocomplete list of stopped containers to select from, or just type in container name then starts the container
  - /ping
    - Replies with "Pong!" when the bot is listening
  - /server
    - Replies with Server Name and member count, good for testing.
