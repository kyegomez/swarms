import requests
import datetime

# Replace these with your repository and token details
GITHUB_TOKEN = "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"
REPO_OWNER = "kyegomez"
REPO_NAME = "swarms"

# GitHub API headers
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

# Get today's date
today = datetime.date.today()

# Fetch closed PRs
def get_closed_prs():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls"
    params = {
        "state": "closed",
        "sort": "updated",
        "direction": "desc",
    }
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    prs = response.json()

    # Filter PRs closed today
    closed_today = [
        pr for pr in prs if pr["closed_at"] and pr["closed_at"].startswith(today.isoformat())
    ]
    return closed_today

# Reopen a PR
def reopen_pr(pr_number):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}"
    data = {"state": "open"}
    response = requests.patch(url, headers=HEADERS, json=data)
    if response.status_code == 200:
        print(f"Successfully reopened PR #{pr_number}")
    else:
        print(f"Failed to reopen PR #{pr_number}: {response.status_code} - {response.text}")

# Main function
def main():
    closed_prs = get_closed_prs()
    if not closed_prs:
        print("No PRs closed today.")
        return
    
    print(f"Found {len(closed_prs)} PR(s) closed today. Reopening them...")
    for pr in closed_prs:
        reopen_pr(pr["number"])

if __name__ == "__main__":
    main()
