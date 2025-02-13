import os
from pathlib import Path
from typing import Optional
from swarms.utils.loguru_logger import initialize_logger


logger = initialize_logger("workspace-manager")


class WorkspaceManager:
    """
    Manages the workspace directory and settings for the application.
    This class is responsible for setting up the workspace directory, logging configuration,
    and retrieving environment variables for telemetry and API key.
    """

    def __init__(
        self,
        workspace_dir: Optional[str] = "agent_workspace",
        use_telemetry: Optional[bool] = True,
        api_key: Optional[str] = None,
    ):
        """
        Initializes the WorkspaceManager with optional parameters for workspace directory,
        telemetry usage, and API key.

        Args:
            workspace_dir (Optional[str]): The path to the workspace directory.
            use_telemetry (Optional[bool]): A flag indicating whether to use telemetry.
            api_key (Optional[str]): The API key for the application.
        """
        self.workspace_dir = workspace_dir
        self.use_telemetry = use_telemetry
        self.api_key = api_key

    def _create_env_file(self, env_file_path: Path) -> None:
        """
        Create a new .env file with default WORKSPACE_DIR.

        Args:
            env_file_path (Path): The path to the .env file.
        """
        with env_file_path.open("w") as file:
            file.write(f"WORKSPACE_DIR={self.workspace_dir}\n")
        logger.info(
            "Created a new .env file with default WORKSPACE_DIR."
        )

    def _append_to_env_file(self, env_file_path: Path) -> None:
        """
        Append WORKSPACE_DIR to .env if it doesn't exist.

        Args:
            env_file_path (Path): The path to the .env file.
        """
        with env_file_path.open("r+") as file:
            content = file.read()
            if "WORKSPACE_DIR" not in content:
                file.seek(0, os.SEEK_END)
                file.write(f"WORKSPACE_DIR={self.workspace_dir}\n")
                logger.info("Appended WORKSPACE_DIR to .env file.")

    def _get_workspace_dir(
        self, workspace_dir: Optional[str] = None
    ) -> str:
        """
        Get the workspace directory from environment variable or default.

        Args:
            workspace_dir (Optional[str]): The path to the workspace directory.

        Returns:
            str: The path to the workspace directory.
        """
        return workspace_dir or os.getenv(
            "WORKSPACE_DIR", "agent_workspace"
        )

    def _get_telemetry_status(
        self, use_telemetry: Optional[bool] = None
    ) -> bool:
        """
        Get telemetry status from environment variable or default.

        Args:
            use_telemetry (Optional[bool]): A flag indicating whether to use telemetry.

        Returns:
            bool: The status of telemetry usage.
        """
        return (
            use_telemetry
            if use_telemetry is not None
            else os.getenv("USE_TELEMETRY", "true").lower() == "true"
        )

    def _get_api_key(
        self, api_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Get API key from environment variable or default.

        Args:
            api_key (Optional[str]): The API key for the application.

        Returns:
            Optional[str]: The API key or None if not set.
        """
        return api_key or os.getenv("SWARMS_API_KEY")

    def _init_workspace(self) -> None:
        """
        Initialize the workspace directory if it doesn't exist.
        """
        if not self.workspace_path.exists():
            self.workspace_path.mkdir(parents=True, exist_ok=True)
        logger.info("Workspace directory initialized.")

    @property
    def get_workspace_path(self) -> Path:
        """
        Get the workspace path.

        Returns:
            Path: The path to the workspace directory.
        """
        return self.workspace_path

    @property
    def get_telemetry_status(self) -> bool:
        """
        Get telemetry status.

        Returns:
            bool: The status of telemetry usage.
        """
        return self.use_telemetry

    @property
    def get_api_key(self) -> Optional[str]:
        """
        Get API key.

        Returns:
            Optional[str]: The API key or None if not set.
        """
        return self.api_key

    def run(self) -> None:
        try:
            # Check if .env file exists and create it if it doesn't
            env_file_path = Path(".env")

            # If the .env file doesn't exist, create it
            if not env_file_path.exists():
                self._create_env_file(env_file_path)
            else:
                # Append WORKSPACE_DIR to .env if it doesn't exist
                self._append_to_env_file(env_file_path)

            # Set workspace directory
            self.workspace_dir = self._get_workspace_dir(
                self.workspace_dir
            )
            self.workspace_path = Path(self.workspace_dir)

            # Set telemetry preference
            self.use_telemetry = self._get_telemetry_status(
                self.use_telemetry
            )

            # Set API key
            self.api_key = self._get_api_key(self.api_key)

            # Initialize workspace
            self._init_workspace()
        except Exception as e:
            logger.error(f"Error initializing WorkspaceManager: {e}")
