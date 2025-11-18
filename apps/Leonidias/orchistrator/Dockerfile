FROM node:19-slim

WORKDIR /app

COPY package.json /app
RUN npm install

COPY . .

ENV DISCORD_TOKEN "" \
    DISCORD_CLIENT_ID "" \
    DISCORD_GUILD_ID ""

CMD [ "node", "index.js" ]
