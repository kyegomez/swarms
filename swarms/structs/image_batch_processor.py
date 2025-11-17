import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from swarms.structs import Agent


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""

    pass


class InvalidAgentError(Exception):
    """Custom exception for invalid agent configurations."""

    pass


class ImageAgentBatchProcessor:
    """
    A class for processing multiple images in parallel using one or more agents.

    This processor can:
    - Handle multiple images from a directory
    - Process images with single or multiple agents
    - Execute tasks in parallel
    - Provide detailed logging and error handling

    Attributes:
        agents (List[Agent]): List of agents to process images
        max_workers (int): Maximum number of parallel workers
        supported_formats (set): Set of supported image formats
    """

    def __init__(
        self,
        agents: Union[Agent, List[Agent], Callable, List[Callable]],
        max_workers: int = None,
        supported_formats: Optional[List[str]] = None,
    ):
        """
        Initialize the ImageBatchProcessor.

        Args:
            agents: Single agent or list of agents to process images
            max_workers: Maximum number of parallel workers (default: 4)
            supported_formats: List of supported image formats (default: ['.jpg', '.jpeg', '.png'])

        Raises:
            InvalidAgentError: If agents parameter is invalid
        """
        self.agents = agents
        self.max_workers = max_workers
        self.supported_formats = supported_formats

        self.agents = (
            [agents] if isinstance(agents, Agent) else agents
        )
        if not self.agents:
            raise InvalidAgentError(
                "At least one agent must be provided"
            )

        # Get 95% of the total number of cores
        self.max_workers = int(os.cpu_count() * 0.95)

        self.supported_formats = set(
            supported_formats or [".jpg", ".jpeg", ".png"]
        )

        # Setup logging
        logger.add(
            "image_processor.log",
            rotation="100 MB",
            retention="10 days",
            level="INFO",
        )

    def _validate_image_path(
        self, image_path: Union[str, Path]
    ) -> Path:
        """
        Validate if the image path exists and has supported format.

        Args:
            image_path: Path to the image file

        Returns:
            Path: Validated Path object

        Raises:
            ImageProcessingError: If path is invalid or format not supported
        """
        path = Path(image_path)
        if not path.exists():
            raise ImageProcessingError(
                f"Image path does not exist: {path}"
            )
        if path.suffix.lower() not in self.supported_formats:
            raise ImageProcessingError(
                f"Unsupported image format {path.suffix}. Supported formats: {self.supported_formats}"
            )
        return path

    def _process_single_image(
        self,
        image_path: Path,
        tasks: Union[str, List[str]],
        agent: Agent,
    ) -> Dict[str, Any]:
        """
        Process a single image with one agent and one or more tasks.

        Args:
            image_path: Path to the image
            tasks: Single task or list of tasks to perform
            agent: Agent to process the image

        Returns:
            Dict containing results for each task
        """
        try:
            tasks_list = [tasks] if isinstance(tasks, str) else tasks
            results = {}

            logger.info(
                f"Processing image {image_path} with agent {agent.__class__.__name__}"
            )
            start_time = time.time()

            for task in tasks_list:
                try:
                    result = agent.run(task=task, img=str(image_path))
                    results[task] = result
                except Exception as e:
                    logger.error(
                        f"Error processing task '{task}' for image {image_path}: {str(e)}"
                    )
                    results[task] = f"Error: {str(e)}"

            processing_time = time.time() - start_time
            logger.info(
                f"Completed processing {image_path} in {processing_time:.2f} seconds"
            )

            return {
                "image_path": str(image_path),
                "results": results,
                "processing_time": processing_time,
            }

        except Exception as e:
            logger.error(
                f"Failed to process image {image_path}: {str(e)}"
            )
            raise ImageProcessingError(
                f"Failed to process image {image_path}: {str(e)}"
            )

    def run(
        self,
        image_paths: Union[str, List[str], Path],
        tasks: Union[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in parallel with the configured agents.

        Args:
            image_paths: Single image path or list of image paths or directory path
            tasks: Single task or list of tasks to perform on each image

        Returns:
            List of dictionaries containing results for each image

        Raises:
            ImageProcessingError: If any image processing fails
        """
        # Handle directory input
        if (
            isinstance(image_paths, (str, Path))
            and Path(image_paths).is_dir()
        ):
            image_paths = [
                os.path.join(image_paths, f)
                for f in os.listdir(image_paths)
                if Path(os.path.join(image_paths, f)).suffix.lower()
                in self.supported_formats
            ]
        elif isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]

        # Validate all paths
        validated_paths = [
            self._validate_image_path(path) for path in image_paths
        ]

        if not validated_paths:
            logger.warning("No valid images found to process")
            return []

        logger.info(
            f"Starting batch processing of {len(validated_paths)} images"
        )
        results = []

        with ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_path = {}

            # Submit all tasks
            for path in validated_paths:
                for agent in self.agents:
                    future = executor.submit(
                        self._process_single_image, path, tasks, agent
                    )
                    future_to_path[future] = (path, agent)

            # Collect results as they complete
            for future in as_completed(future_to_path):
                path, agent = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Failed to process {path} with {agent.__class__.__name__}: {str(e)}"
                    )
                    results.append(
                        {
                            "image_path": str(path),
                            "error": str(e),
                            "agent": agent.__class__.__name__,
                        }
                    )

        logger.info(
            f"Completed batch processing of {len(validated_paths)} images"
        )
        return results

    def __call__(self, *args, **kwargs):
        """
        Make the ImageAgentBatchProcessor callable like a function.

        This allows the processor to be used directly as a function, which will
        call the run() method with the provided arguments.

        Args:
            *args: Variable length argument list to pass to run()
            **kwargs: Arbitrary keyword arguments to pass to run()

        Returns:
            The result of calling run() with the provided arguments
        """
        return self.run(*args, **kwargs)
