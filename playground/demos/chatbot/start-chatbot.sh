sudo apt-get install redis-stack-server
sudo systemctl enable redis-stack-server
sudo systemctl start redis-stack-server
cd ~/firecrawl/apps/api
pnpm run workers
pnpm run start