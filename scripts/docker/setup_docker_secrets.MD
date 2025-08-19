# Setting up DockerHub Secrets for GitHub Actions

This guide will help you set up the required secrets for the Docker workflow to automatically build and push images to DockerHub.

## Prerequisites

1. A DockerHub account
2. Admin access to the GitHub repository
3. DockerHub access token

## Step 1: Create a DockerHub Access Token

1. Log in to [DockerHub](https://hub.docker.com/)
2. Go to your account settings
3. Navigate to "Security" → "Access Tokens"
4. Click "New Access Token"
5. Give it a name (e.g., "GitHub Actions")
6. Set the permissions to "Read & Write"
7. Copy the generated token (you won't be able to see it again!)

## Step 2: Add Secrets to GitHub Repository

1. Go to your GitHub repository
2. Navigate to "Settings" → "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Add the following secrets:

### Required Secrets

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `DOCKERHUB_USERNAME` | Your DockerHub username | Your DockerHub username (e.g., `kyegomez`) |
| `DOCKERHUB_TOKEN` | Your DockerHub access token | The access token you created in Step 1 |

## Step 3: Verify Setup

1. Push a commit to the `main` branch
2. Go to the "Actions" tab in your GitHub repository
3. You should see the "Docker Build and Publish" workflow running
4. Check that it completes successfully

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   - Double-check your DockerHub username and token
   - Ensure the token has "Read & Write" permissions
   - Make sure the token hasn't expired

2. **Permission Denied**
   - Verify you have admin access to the repository
   - Check that the secrets are named exactly as shown above

3. **Workflow Not Triggering**
   - Ensure you're pushing to the `main` branch
   - Check that the workflow file is in `.github/workflows/`
   - Verify the workflow file has the correct triggers

### Testing Locally

You can test the Docker build locally before pushing:

```bash
# Build the image locally
docker build -t swarms:test .

# Test the image
docker run --rm swarms:test python test_docker.py

# If everything works, push to GitHub
git add .
git commit -m "Add Docker support"
git push origin main
```

## Security Notes

- Never commit secrets directly to your repository
- Use repository secrets for sensitive information
- Regularly rotate your DockerHub access tokens
- Consider using organization-level secrets for team repositories

## Additional Configuration

### Custom Registry

If you want to use a different registry (not DockerHub), update the workflow file:

```yaml
env:
  REGISTRY: your-registry.com
  IMAGE_NAME: your-org/your-repo
```

### Multiple Tags

The workflow automatically creates tags based on:
- Git branch name
- Git commit SHA
- Version tags (v*.*.*)
- Latest tag for main branch

You can customize this in the workflow file under the "Extract Docker metadata" step.

## Support

If you encounter issues:

1. Check the GitHub Actions logs for detailed error messages
2. Verify your DockerHub credentials
3. Ensure the workflow file is properly configured
4. Open an issue in the repository with the error details 
