"""
Image file utilities for converting images to base64 data URIs.

This module provides functions for handling various image input formats and
converting them to base64-encoded data URIs suitable for use with LLM APIs.
"""

import base64
import re
import uuid
from pathlib import Path

import requests
from loguru import logger


def is_base64_encoded(image_source: str) -> bool:
    """
    Check if a string is already base64 encoded (either as data URI or raw base64).

    Args:
        image_source (str): The string to check.

    Returns:
        bool: True if the string appears to be base64 encoded, False otherwise.
    """
    # Check if it's a data URI
    if image_source.startswith("data:image"):
        return True

    # Check if it's a raw base64 string (no data URI prefix)
    # Base64 strings are typically long and contain only base64 characters
    # We check for reasonable length and base64 character set
    if len(image_source) > 100:  # Base64 images are typically long
        try:
            # Try to decode a sample to verify it's valid base64
            base64.b64decode(image_source[:100] + "==", validate=True)
            # Check if it contains only base64 characters (A-Z, a-z, 0-9, +, /, =)
            base64_pattern = re.compile(r"^[A-Za-z0-9+/=]+$")
            if base64_pattern.match(image_source):
                return True
        except Exception:
            pass

    return False


def get_image_base64(image_source: str) -> str:
    """
    Convert image data from a URL, local file path, data URI, or raw base64 string to a base64-encoded string in data URI format.

    If the input is already a data URI, it is returned unchanged. If it's a raw base64 string (without data URI prefix),
    it is converted to a data URI format. Otherwise, the image is loaded from the specified source, encoded as base64,
    and returned as a data URI with the appropriate MIME type.

    Args:
        image_source (str): The path, URL, data URI, or raw base64-encoded string of the image.

    Returns:
        str: The image as a base64-encoded data URI string.

    Raises:
        requests.HTTPError: If fetching the image from a URL fails.
        FileNotFoundError: If the local image file does not exist.
        ValueError: If the base64 string is invalid.

    Example:
        >>> # From file path
        >>> uri = get_image_base64("example.jpg")
        >>> print(uri[:40])
        data:image/jpeg;base64,/9j/4AAQSkZJRg...

        >>> # From URL
        >>> uri = get_image_base64("https://example.com/image.png")

        >>> # Already a data URI (returns as-is)
        >>> uri = get_image_base64("data:image/png;base64,iVBORw0KGgo...")
    """
    # If already a data URI, return as-is
    if image_source.startswith("data:image"):
        return image_source

    # Check if it's a raw base64 string (without data URI prefix)
    if is_base64_encoded(
        image_source
    ) and not image_source.startswith(("http://", "https://")):
        # It's a raw base64 string, convert to data URI format
        # Default to JPEG if we can't determine the format
        # In practice, users should provide data URI format, but we support raw base64 for flexibility
        mime_type = "image/jpeg"  # Default MIME type
        return f"data:{mime_type};base64,{image_source}"

    # Handle URLs
    if image_source.startswith(("http://", "https://")):
        response = requests.get(image_source)
        response.raise_for_status()
        image_data = response.content
    else:
        # Assume it's a file path
        with open(image_source, "rb") as file:
            image_data = file.read()

    extension = Path(image_source).suffix.lower()
    mime_type_mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".svg": "image/svg+xml",
    }
    mime_type = mime_type_mapping.get(extension, "image/jpeg")
    encoded_string = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"


def get_image_data_uri(image_path: str) -> str:
    """
    Reads an image file and returns a data URI string representing the base64-encoded image.

    This is a convenience wrapper around `get_image_base64` for backward compatibility.
    It only handles local file paths, while `get_image_base64` supports URLs, data URIs,
    and raw base64 strings as well.

    Supports common image formats including JPEG, PNG, GIF, WebP, BMP, TIFF, and SVG.
    The MIME type is automatically detected from the file extension.

    Args:
        image_path (str): The file path to the image.

    Returns:
        str: A data URI string in the format "data:image/<type>;base64,<base64-data>".
            The MIME type is determined from the file extension (e.g., "data:image/png;base64,..."
            for PNG files, "data:image/jpeg;base64,..." for JPEG files).

    Example:
        >>> uri = get_image_data_uri("example.jpg")
        >>> print(uri[:40])
        data:image/jpeg;base64,/9j/4AAQSkZJRg...
        >>> uri = get_image_data_uri("example.png")
        >>> print(uri[:40])
        data:image/png;base64,iVBORw0KGgoAAAANS...

    Note:
        For more flexibility (URLs, data URIs, raw base64), use `get_image_base64` instead.
    """
    return get_image_base64(image_path)


def save_base64_as_image(
    base64_data: str,
    output_dir: str = "images",
) -> str:
    """
    Decode base64-encoded image data and save it as an image file in the specified directory.

    This function supports both raw base64 strings and data URIs (data:image/...;base64,...).
    The image format is determined from the MIME type if present, otherwise defaults to JPEG.
    The image is saved with a randomly generated filename.

    Args:
        base64_data (str): The base64-encoded image data, either as a raw string or a data URI.
        output_dir (str, optional): Directory to save the image file. Defaults to "images".
            If None, saves to the current working directory.

    Returns:
        str: The full path to the saved image file.

    Raises:
        ValueError: If the base64 data is not a valid data URI or is otherwise invalid.
        IOError: If the image cannot be written to disk.
    """
    import os

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if base64_data.startswith("data:image"):
        try:
            header, encoded_data = base64_data.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
        except (ValueError, IndexError):
            raise ValueError("Invalid data URI format")
    else:
        encoded_data = base64_data
        mime_type = "image/jpeg"

    mime_to_extension = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/svg+xml": ".svg",
    }
    extension = mime_to_extension.get(mime_type, ".jpg")
    filename = f"{uuid.uuid4()}{extension}"
    file_path = os.path.join(output_dir, filename)

    try:
        logger.debug(
            f"Attempting to decode base64 data of length: {len(encoded_data)}"
        )
        logger.debug(
            f"Base64 data (first 100 chars): {encoded_data[:100]}..."
        )
        image_data = base64.b64decode(encoded_data)
        with open(file_path, "wb") as f:
            f.write(image_data)
        logger.info(f"Image saved successfully to: {file_path}")
        return file_path
    except Exception as e:
        logger.error(
            f"Base64 decoding failed. Data length: {len(encoded_data)}"
        )
        logger.error(
            f"First 100 chars of data: {encoded_data[:100]}..."
        )
        raise IOError(f"Failed to save image: {str(e)}")
