# Hosting Agents on Google Cloud Run

This documentation provides a highly detailed, step-by-step guide to hosting your agents using Google Cloud Run. It uses a well-structured project setup that includes a Dockerfile at the root level, a folder dedicated to your API file, and a `requirements.txt` file to manage all dependencies. This guide will ensure your deployment is scalable, efficient, and easy to maintain.

---

## **Project Structure**

Your project directory should adhere to the following structure to ensure compatibility and ease of deployment:

```
.
├── Dockerfile
├── requirements.txt
└── api/
    └── api.py
```

Each component serves a specific purpose in the deployment pipeline, ensuring modularity and maintainability.

---

## **Step 1: Prerequisites**

Before you begin, make sure to satisfy the following prerequisites to avoid issues during deployment:

1. **Google Cloud Account**:
   - Create a Google Cloud account at [Google Cloud Console](https://console.cloud.google.com/).
   - Enable billing for your project. Billing is necessary for accessing Cloud Run services.

2. **Install Google Cloud SDK**:
   - Follow the [installation guide](https://cloud.google.com/sdk/docs/install) to set up the Google Cloud SDK on your local machine.

3. **Install Docker**:
   - Download and install Docker by following the [official Docker installation guide](https://docs.docker.com/get-docker/). Docker is crucial for containerizing your application.

4. **Create a Google Cloud Project**:
   - Navigate to the Google Cloud Console and create a new project. Assign it a meaningful name and note the **Project ID**, as it will be used throughout this guide.

5. **Enable Required APIs**:
   - Visit the [API Library](https://console.cloud.google.com/apis/library) and enable the following APIs:
     - Cloud Run API
     - Cloud Build API
     - Artifact Registry API
   - These APIs are essential for deploying and managing your application in Cloud Run.

---

## **Step 2: Creating the Files**

### 1. **`api/api.py`**

This is the main Python script where you define your Swarms agents and expose an API endpoint for interacting with them. Here’s an example:

```python
from flask import Flask, request, jsonify
from swarms import Agent  # Assuming `swarms` is the framework you're using

app = Flask(__name__)

# Example Swarm agent
agent = Agent(
    agent_name="Stock-Analysis-Agent",
    model_name="gpt-4o-mini",
    max_loops="auto",
    interactive=True,
    streaming_on=True,
)

@app.route('/run-agent', methods=['POST'])
def run_agent():
    data = request.json
    task = data.get('task', '')
    result = agent.run(task)
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

This example sets up a basic API that listens for POST requests, processes a task using a Swarm agent, and returns the result as a JSON response. Customize it based on your agent’s functionality.

---

### 2. **`requirements.txt`**

This file lists all Python dependencies required for your project. Example:

```
flask
swarms
# add any other dependencies here
```

Be sure to include any additional libraries your agents rely on. Keeping this file up to date ensures smooth dependency management during deployment.

---

### 3. **`Dockerfile`**

The Dockerfile specifies how your application is containerized. Below is a sample Dockerfile for your setup:

```dockerfile
# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY api/ ./api/

# Expose port 8080 (Cloud Run default port)
EXPOSE 8080

# Run the application
CMD ["python", "api/api.py"]
```

This Dockerfile ensures your application is containerized with minimal overhead, focusing on slim images for efficiency.

---

## **Step 3: Deploying to Google Cloud Run**

### 1. **Authenticate with Google Cloud**

Log in to your Google Cloud account by running:

```bash
gcloud auth login
```

Set the active project to match your deployment target:

```bash
gcloud config set project [PROJECT_ID]
```

Replace `[PROJECT_ID]` with your actual Project ID.

---

### 2. **Build the Docker Image**

Use Google Cloud's Artifact Registry to store and manage your Docker image. Follow these steps:

1. **Create a Repository**:

```bash
gcloud artifacts repositories create my-repo --repository-format=Docker --location=us-central1
```

2. **Authenticate Docker with Google Cloud**:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

3. **Build and Tag the Image**:

```bash
docker build -t us-central1-docker.pkg.dev/[PROJECT_ID]/my-repo/my-image .
```

4. **Push the Image**:

```bash
docker push us-central1-docker.pkg.dev/[PROJECT_ID]/my-repo/my-image
```

---

### 3. **Deploy to Cloud Run**

Deploy the application to Cloud Run with the following command:

```bash
gcloud run deploy my-agent-service \
  --image us-central1-docker.pkg.dev/[PROJECT_ID]/my-repo/my-image \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

Key points:
- Replace `[PROJECT_ID]` with your actual Project ID.
- The `--allow-unauthenticated` flag makes the service publicly accessible. Exclude it to restrict access.

---

## **Step 4: Testing the Deployment**

Once the deployment is complete, test the service:

1. Note the URL provided by Cloud Run.
2. Use `curl` or Postman to send a request. Example:

```bash
curl -X POST [CLOUD_RUN_URL]/run-agent \
  -H "Content-Type: application/json" \
  -d '{"task": "example task"}'
```

This tests whether your agent processes the task correctly and returns the expected output.

---

## **Step 5: Updating the Service**

To apply changes to your application:

1. Edit the necessary files.
2. Rebuild and push the updated Docker image:

```bash
docker build -t us-central1-docker.pkg.dev/[PROJECT_ID]/my-repo/my-image .
docker push us-central1-docker.pkg.dev/[PROJECT_ID]/my-repo/my-image
```

3. Redeploy the service:

```bash
gcloud run deploy my-agent-service \
  --image us-central1-docker.pkg.dev/[PROJECT_ID]/my-repo/my-image
```

This ensures the latest version of your application is live.

---

## **Troubleshooting**

- **Permission Errors**:
  Ensure your account has roles like Cloud Run Admin and Artifact Registry Reader.
- **Port Issues**:
  Confirm the application listens on port 8080. Cloud Run expects this port by default.
- **Logs**:
  Use the Google Cloud Console or CLI to review logs for debugging:

```bash
gcloud logs read --project [PROJECT_ID]
```

---

## **Conclusion**

By following this comprehensive guide, you can deploy your agents on Google Cloud Run with ease. This method leverages Docker for containerization and Google Cloud services for seamless scalability and management. With a robust setup like this, you can focus on enhancing your agents’ capabilities rather than worrying about deployment challenges.

