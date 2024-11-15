import os
from pathlib import Path
from loguru import logger

class WorkspaceManager:
    DEFAULT_WORKSPACE = "agent_workspace"
    
    @classmethod
    def get_workspace_dir(cls) -> str:
        """Get or create workspace directory with proper fallback"""
        workspace = os.getenv("WORKSPACE_DIR", cls.DEFAULT_WORKSPACE)
        workspace_path = Path(workspace)
        
        try:
            # Create directory if it doesn't exist
            workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Set environment variable if not already set
            if "WORKSPACE_DIR" not in os.environ:
                os.environ["WORKSPACE_DIR"] = str(workspace_path)
                
            return str(workspace_path)
        except Exception as e:
            logger.error(f"Error creating workspace directory: {e}")
            raise
