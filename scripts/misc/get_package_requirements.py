import pkg_resources


def get_package_versions(requirements_path, output_path):
    try:
        with open(requirements_path) as file:
            requirements = file.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{requirements_path}' was not found.")
        return

    package_versions = []

    for requirement in requirements:
        # Skip empty lines and comments
        if (
            requirement.strip() == ""
            or requirement.strip().startswith("#")
        ):
            continue

        # Extract package name
        package_name = requirement.split("==")[0].strip()
        try:
            version = pkg_resources.get_distribution(
                package_name
            ).version
            package_versions.append(f"{package_name}=={version}")
        except pkg_resources.DistributionNotFound:
            package_versions.append(f"{package_name}: not installed")

    with open(output_path, "w") as file:
        for package_version in package_versions:
            file.write(package_version + "\n")
    print(f"Versions written to {output_path}")


# Usage
get_package_versions("requirements.txt", "installed_versions.txt")
