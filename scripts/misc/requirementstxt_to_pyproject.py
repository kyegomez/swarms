import pkg_resources
import toml


def update_pyproject_versions(pyproject_path):
    try:
        with open(pyproject_path) as file:
            data = toml.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{pyproject_path}' was not found.")
        return
    except toml.TomlDecodeError:
        print(
            f"Error: The file '{pyproject_path}' is not a valid TOML"
            " file."
        )
        return

    dependencies = (
        data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    )

    for package in dependencies:
        if package.lower() == "python":
            continue  # Skip the Python version dependency

        try:
            version = pkg_resources.get_distribution(package).version
            dependencies[package] = version
        except pkg_resources.DistributionNotFound:
            print(f"Warning: Package '{package}' not installed.")

    with open(pyproject_path, "w") as file:
        toml.dump(data, file)

    print(f"Updated versions written to {pyproject_path}")


# Usage
update_pyproject_versions("pyproject.toml")
