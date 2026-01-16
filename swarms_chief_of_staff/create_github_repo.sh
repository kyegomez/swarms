#!/usr/bin/env bash
# Simple helper to create a GitHub repo using gh CLI and push current directory
set -e
REPO_NAME=${1:-swarms-chief-of-staff}
OWNER=${2:-$(gh api user --jq .login 2>/dev/null || echo '')}
if [ -z "$OWNER" ]; then
  echo "gh CLI not authenticated or not installed. Authenticate with 'gh auth login' or pass owner as second arg." >&2
  exit 1
fi

echo "Creating repo $OWNER/$REPO_NAME..."
gh repo create "$OWNER/$REPO_NAME" --public --confirm --source . --remote origin

echo "Pushed. You can now visit: https://github.com/$OWNER/$REPO_NAME"
