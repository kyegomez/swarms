import platform
import subprocess

import pkg_resources
import psutil
import toml


def get_python_version():
    return platform.python_version()


def get_pip_version():
    try:
        pip_version = (
            subprocess.check_output(["pip", "--version"])
            .decode()
            .split()[1]
        )
    except Exception as e:
        pip_version = str(e)
    return pip_version


def get_swarms_verison():
    try:
        swarms_verison_cmd = (
            subprocess.check_output(["swarms", "--version"])
            .decode()
            .split()[1]
        )
    except Exception as e:
        swarms_verison_cmd = str(e)
    swarms_verison_pkg = pkg_resources.get_distribution(
        "swarms"
    ).version
    swarms_verison = swarms_verison_cmd, swarms_verison_pkg
    return swarms_verison


def get_os_version():
    return platform.platform()


def get_cpu_info():
    return platform.processor()


def get_ram_info():
    vm = psutil.virtual_memory()
    used_ram_gb = vm.used / (1024**3)
    free_ram_gb = vm.free / (1024**3)
    total_ram_gb = vm.total / (1024**3)
    return (
        f"{total_ram_gb:.2f} GB, used: {used_ram_gb:.2f}, free:"
        f" {free_ram_gb:.2f}"
    )


def get_package_mismatches(file_path="pyproject.toml"):
    with open(file_path) as file:
        pyproject = toml.load(file)
    dependencies = pyproject["tool"]["poetry"]["dependencies"]
    dev_dependencies = pyproject["tool"]["poetry"]["group"]["dev"][
        "dependencies"
    ]
    dependencies.update(dev_dependencies)

    installed_packages = {
        pkg.key: pkg.version for pkg in pkg_resources.working_set
    }

    mismatches = []
    for package, version_info in dependencies.items():
        if isinstance(version_info, dict):
            version_info = version_info["version"]
        installed_version = installed_packages.get(package)
        if installed_version and version_info.startswith("^"):
            expected_version = version_info[1:]
            if not installed_version.startswith(expected_version):
                mismatches.append(
                    f"\t  {package}: Mismatch,"
                    f" pyproject.toml={expected_version},"
                    f" pip={installed_version}"
                )
        else:
            mismatches.append(f"\t  {package}: Not found in pip list")

    return "\n" + "\n".join(mismatches)


def system_info():
    return {
        "Python Version": get_python_version(),
        "Pip Version": get_pip_version(),
        # "Swarms Version": swarms_verison,
        "OS Version and Architecture": get_os_version(),
        "CPU Info": get_cpu_info(),
        "RAM Info": get_ram_info(),
    }
