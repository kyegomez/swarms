## To run project in Phala TEE

1. Build the docker image and publish it to the docker hub
    `docker compose build -t <your-dockerhub-username>/swarm-agent-node:latest`
    `docker push <your-dockerhub-username>/swarm-agent-node:latest`
2. Deploy to Phala cloud using [tee-cloud-cli](https://github.com/Phala-Network/tee-cloud-cli) or manually with the [Cloud dashboard](https://cloud.phala.network/).
3. Check your agent's TEE proof and verify it on the [TEE Attestation Explorer](https://proof.t16z.com/).
