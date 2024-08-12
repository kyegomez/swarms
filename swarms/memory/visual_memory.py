from datetime import datetime
from typing import List


class VisualShortTermMemory:
    """
    A class representing visual short-term memory.

    Attributes:
        memory (list): A list to store images and their descriptions.

    Examples:
        example = VisualShortTermMemory()
        example.add(
            images=["image1.jpg", "image2.jpg"],
            description=["description1", "description2"],
            timestamps=[1.0, 2.0],
            locations=["location1", "location2"],
        )
        print(example.return_as_string())
        # print(example.get_images())
    """

    def __init__(self):
        self.memory = []

    def add(
        self,
        images: List[str] = None,
        description: List[str] = None,
        timestamps: List[float] = None,
        locations: List[str] = None,
    ):
        """
        Add images and their descriptions to the memory.

        Args:
            images (list): A list of image paths.
            description (list): A list of corresponding descriptions.
            timestamps (list): A list of timestamps for each image.
            locations (list): A list of locations where the images were captured.
        """
        current_time = datetime.now()

        # Create a dictionary of each image and description
        # and append it to the memory
        for image, description, timestamp, location in zip(
            images, description, timestamps, locations
        ):
            self.memory.append(
                {
                    "image": image,
                    "description": description,
                    "timestamp": timestamp,
                    "location": location,
                    "added_at": current_time,
                }
            )

    def get_images(self):
        """
        Get a list of all images in the memory.

        Returns:
            list: A list of image paths.
        """
        return [item["image"] for item in self.memory]

    def get_descriptions(self):
        """
        Get a list of all descriptions in the memory.

        Returns:
            list: A list of descriptions.
        """
        return [item["description"] for item in self.memory]

    def search_by_location(self, location: str):
        """
        Search for images captured at a specific location.

        Args:
            location (str): The location to search for.

        Returns:
            list: A list of images captured at the specified location.
        """
        return [
            item["image"]
            for item in self.memory
            if item["location"] == location
        ]

    def search_by_timestamp(self, start_time: float, end_time: float):
        """
        Search for images captured within a specific time range.

        Args:
            start_time (float): The start time of the range.
            end_time (float): The end time of the range.

        Returns:
            list: A list of images captured within the specified time range.
        """
        return [
            item["image"]
            for item in self.memory
            if start_time <= item["timestamp"] <= end_time
        ]

    def return_as_string(self):
        """
        Return the memory as a string.

        Returns:
            str: A string representation of the memory.
        """
        return str(self.memory)
