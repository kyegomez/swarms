from swarms.utils.auto_download_check_packages import auto_check_and_download_package, check_and_install_package

def test_check_and_install_package_pip():
    result = check_and_install_package("numpy", package_manager="pip")
    print(f"Test result for 'numpy' installation using pip: {result}")
    assert result, "Failed to install or verify 'numpy' using pip"

def test_check_and_install_package_conda():
    result = check_and_install_package("numpy", package_manager="conda")
    print(f"Test result for 'numpy' installation using conda: {result}")
    assert result, "Failed to install or verify 'numpy' using conda"

def test_check_and_install_specific_version():
    result = check_and_install_package("numpy", package_manager="pip", version="1.21.0")
    print(f"Test result for specific version of 'numpy' installation using pip: {result}")
    assert result, "Failed to install or verify specific version of 'numpy' using pip"

def test_check_and_install_with_upgrade():
    result = check_and_install_package("numpy", package_manager="pip", upgrade=True)
    print(f"Test result for 'numpy' upgrade using pip: {result}")
    assert result, "Failed to upgrade 'numpy' using pip"

def test_auto_check_and_download_single_package():
    result = auto_check_and_download_package("scipy", package_manager="pip")
    print(f"Test result for 'scipy' installation using pip: {result}")
    assert result, "Failed to install or verify 'scipy' using pip"

def test_auto_check_and_download_multiple_packages():
    packages = ["scipy", "pandas"]
    result = auto_check_and_download_package(packages, package_manager="pip")
    print(f"Test result for multiple packages installation using pip: {result}")
    assert result, f"Failed to install or verify one or more packages in {packages} using pip"

def test_auto_check_and_download_multiple_packages_with_versions():
    packages = ["numpy:1.21.0", "pandas:1.3.0"]
    result = auto_check_and_download_package(packages, package_manager="pip")
    print(f"Test result for multiple packages with versions installation using pip: {result}")
    assert result, f"Failed to install or verify one or more packages in {packages} with specific versions using pip"

# Example of running tests
if __name__ == "__main__":
    try:
        test_check_and_install_package_pip()
        print("test_check_and_install_package_pip passed")
        
        test_check_and_install_package_conda()
        print("test_check_and_install_package_conda passed")
        
        test_check_and_install_specific_version()
        print("test_check_and_install_specific_version passed")
        
        test_check_and_install_with_upgrade()
        print("test_check_and_install_with_upgrade passed")
        
        test_auto_check_and_download_single_package()
        print("test_auto_check_and_download_single_package passed")
        
        test_auto_check_and_download_multiple_packages()
        print("test_auto_check_and_download_multiple_packages passed")
        
        test_auto_check_and_download_multiple_packages_with_versions()
        print("test_auto_check_and_download_multiple_packages_with_versions passed")
        
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
