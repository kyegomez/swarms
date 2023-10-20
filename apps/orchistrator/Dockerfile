FROM node:19-slim

WORKDIR /app

COPY package.json /app
RUN npm install

COPY . .

CMD [ "node", "index.js" ]
