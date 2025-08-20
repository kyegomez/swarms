#!/usr/bin/env python3
"""
Test script to verify Swarms installation in Docker container.
"""

import sys
from typing import Dict, Any


def test_swarms_import() -> Dict[str, Any]:
    """
    Test that swarms can be imported and basic functionality works.

    Returns:
        Dict[str, Any]: Test results
    """
    try:
        import swarms

        print(
            f" Swarms imported successfully. Version: {swarms.__version__}"
        )

        # Test basic functionality
        from swarms import Agent

        print(" Agent class imported successfully")

        return {
            "status": "success",
            "version": swarms.__version__,
            "message": "Swarms package is working correctly",
        }

    except ImportError as e:
        print(f" Failed to import swarms: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Swarms package import failed",
        }
    except Exception as e:
        print(f" Unexpected error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Unexpected error occurred",
        }


def main() -> None:
    """Main function to run tests."""
    print(" Testing Swarms Docker Image...")
    print("=" * 50)

    # Test Python version
    print(f"Python version: {sys.version}")

    # Test swarms import
    result = test_swarms_import()

    print("=" * 50)
    if result["status"] == "success":
        print(" All tests passed! Docker image is working correctly.")
        sys.exit(0)
    else:
        print(" Tests failed! Please check the Docker image.")
        sys.exit(1)


if __name__ == "__main__":
    main()
