import requests


def get_example_py_urls():
    owner = "kyegomez"
    repo = "swarms"
    branch = "master"
    examples_path = "examples"

    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    raw_base = (
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/"
    )

    response = requests.get(api_url)
    response.raise_for_status()

    data = response.json()
    all_files = data.get("tree", [])

    example_files = [
        raw_base + file["path"]
        for file in all_files
        if file["path"].startswith(examples_path)
        and file["path"].endswith("example.py")
        and file["type"] == "blob"
    ]

    return example_files


if __name__ == "__main__":
    urls = get_example_py_urls()
    for url in urls:
        print(url)
