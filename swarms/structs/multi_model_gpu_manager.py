"""
GPU Model Manager
================

A production-grade utility for managing multiple PyTorch or Hugging Face models
across available GPUs. This module automatically calculates model memory requirements,
allocates models to appropriate GPUs, and provides a unified interface for running
inference tasks across all loaded models.

Features:
- Dynamic model memory calculation
- Optimal GPU memory allocation
- Multi-processing support for parallel model execution
- Customizable task execution for specific models
- Comprehensive logging and error handling
"""

import os
import queue
import sys
import time
import json
import uuid
import torch
import multiprocessing
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from loguru import logger
import numpy as np
from contextlib import contextmanager

# Try to import transformers, but don't fail if not available
try:
    import transformers
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "Transformers package not found. HuggingFace models will not be supported."
    )


class ModelType(Enum):
    """Enum defining supported model types."""

    PYTORCH = "pytorch"
    HUGGINGFACE = "huggingface"
    UNKNOWN = "unknown"


class GPUAllocationStrategy(Enum):
    """Enum defining GPU allocation strategies."""

    FILL_GPU = "fill_gpu"  # Fill each GPU before moving to next
    DISTRIBUTE = "distribute"  # Distribute models evenly across GPUs
    MEMORY_OPTIMIZED = "memory_optimized"  # Optimize for memory usage


@dataclass
class ModelMetadata:
    """Data class for storing model metadata."""

    name: str
    model_type: ModelType
    memory_required: float  # in GB
    model: Any
    device: Optional[torch.device] = None
    process: Optional[multiprocessing.Process] = None
    loaded: bool = False


@dataclass
class GPUMetadata:
    """Data class for storing GPU metadata."""

    id: int
    device: torch.device
    total_memory: float  # in GB
    available_memory: float  # in GB
    models: List[str] = None

    def __post_init__(self):
        if self.models is None:
            self.models = []


class ModelMemoryCalculator:
    """Utility class for calculating model memory requirements."""

    @staticmethod
    def get_pytorch_model_size(model: torch.nn.Module) -> float:
        """
        Calculate the memory size of a PyTorch model in GB.

        Args:
            model: PyTorch model object

        Returns:
            Memory size in GB
        """
        try:
            # Get model size in parameters
            model_parameters = sum(
                p.numel() for p in model.parameters()
            )

            # Calculate size based on dtype (default to float32)
            if any(
                p.dtype == torch.float16 for p in model.parameters()
            ):
                bytes_per_param = 2  # float16
            elif any(
                p.dtype == torch.bfloat16 for p in model.parameters()
            ):
                bytes_per_param = 2  # bfloat16
            elif any(
                p.dtype == torch.float64 for p in model.parameters()
            ):
                bytes_per_param = 8  # float64
            else:
                bytes_per_param = 4  # float32

            # Calculate raw model size in bytes
            model_size_bytes = model_parameters * bytes_per_param

            # Add 20% for optimizer states, gradients, and other overhead
            model_size_bytes_with_overhead = model_size_bytes * 1.2

            # Convert to GB
            model_size_gb = model_size_bytes_with_overhead / (1024**3)

            # Add a safety margin of 10%
            model_size_gb_with_safety = model_size_gb * 1.1

            return model_size_gb_with_safety

        except Exception as e:
            logger.error(
                f"Error calculating PyTorch model size: {str(e)}"
            )
            # Fallback estimation
            return 2.0  # Default estimation if calculation fails

    @staticmethod
    def get_huggingface_model_size(
        model_or_path: Union[str, Any],
    ) -> float:
        """
        Calculate the memory size of a Hugging Face model in GB.
        Works with either model path or loaded model.

        Args:
            model_or_path: Hugging Face model object or path to model

        Returns:
            Memory size in GB
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error(
                "Transformers package not available. Cannot calculate Hugging Face model size."
            )
            return 5.0  # Default fallback

        try:
            # If it's a path, we'll try to estimate without loading
            if isinstance(model_or_path, str):
                path = Path(model_or_path)
                if path.exists():
                    # Check for model info in config
                    config_path = path / "config.json"
                    if config_path.exists():
                        with open(config_path, "r") as f:
                            config = json.load(f)
                            if "n_params" in config:
                                n_params = config["n_params"]
                                # Estimate with overhead
                                model_size_gb = (
                                    n_params * 4 * 1.5
                                ) / (1024**3)
                                return model_size_gb

                    # Alternatively, estimate from model files
                    pytorch_files = list(path.glob("*.bin"))
                    if pytorch_files:
                        total_size = sum(
                            f.stat().st_size for f in pytorch_files
                        )
                        model_size_gb = (total_size * 1.5) / (
                            1024**3
                        )  # 50% overhead
                        return model_size_gb

                # If we can't estimate, load the model and calculate
                logger.info(
                    f"Loading model from {model_or_path} to calculate memory requirements..."
                )
                model = AutoModel.from_pretrained(model_or_path)
                return ModelMemoryCalculator.get_pytorch_model_size(
                    model
                )
            else:
                # If we already have the model loaded, calculate directly
                return ModelMemoryCalculator.get_pytorch_model_size(
                    model_or_path
                )

        except Exception as e:
            logger.error(
                f"Error calculating Hugging Face model size: {str(e)}"
            )
            return 5.0  # Default estimation if calculation fails


class GPUManager:
    """Manages available GPUs and their memory."""

    def __init__(self):
        """Initialize the GPU manager."""
        self.gpus: List[GPUMetadata] = []
        self._initialize_gpus()

    def _initialize_gpus(self) -> None:
        """
        Initialize available GPUs and collect their metadata.
        """
        if not torch.cuda.is_available():
            logger.warning("No CUDA-capable devices detected.")
            return

        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} CUDA-capable devices.")

        for gpu_id in range(num_gpus):
            device = torch.device(f"cuda:{gpu_id}")

            # Get total memory
            total_memory = torch.cuda.get_device_properties(
                gpu_id
            ).total_memory
            total_memory_gb = total_memory / (1024**3)

            # Get available memory
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
            available_memory = torch.cuda.mem_get_info(device)[0]
            available_memory_gb = available_memory / (1024**3)

            # Create GPU metadata
            gpu_metadata = GPUMetadata(
                id=gpu_id,
                device=device,
                total_memory=total_memory_gb,
                available_memory=available_memory_gb,
            )

            self.gpus.append(gpu_metadata)
            logger.info(
                f"GPU {gpu_id}: {total_memory_gb:.2f} GB total, {available_memory_gb:.2f} GB available"
            )

    def update_gpu_memory_info(self) -> None:
        """
        Update the available memory information for all GPUs.
        """
        if not self.gpus:
            logger.warning(
                "No GPUs available to update memory information."
            )
            return

        for gpu in self.gpus:
            torch.cuda.set_device(gpu.device)
            torch.cuda.empty_cache()
            available_memory = torch.cuda.mem_get_info(gpu.device)[0]
            gpu.available_memory = available_memory / (1024**3)
            logger.debug(
                f"Updated GPU {gpu.id}: {gpu.available_memory:.2f} GB available"
            )


class ModelGrid:
    """
    Main class for managing multiple models across available GPUs.

    This class handles:
    - Loading and unloading models
    - Allocating models to appropriate GPUs based on memory requirements
    - Running inference tasks on specific models
    - Managing model lifecycle through multiple processes
    """

    def __init__(
        self,
        allocation_strategy: GPUAllocationStrategy = GPUAllocationStrategy.MEMORY_OPTIMIZED,
        memory_buffer: float = 0.5,  # GB buffer to leave on each GPU
        max_cpu_models: int = 0,  # Maximum models to keep on CPU if no GPU space
        use_multiprocessing: bool = True,
        log_level: str = "INFO",
    ):
        """
        Initialize the model manager.

        Args:
            allocation_strategy: Strategy for allocating models to GPUs
            memory_buffer: Memory buffer to leave on each GPU (in GB)
            max_cpu_models: Maximum number of models to keep on CPU if no GPU space
            use_multiprocessing: Whether to use multiprocessing for model execution
            log_level: Logging level
        """
        # Set log level
        logger.remove()
        logger.add(sys.stderr, level=log_level)
        logger.add(
            "gpu_model_manager.log",
            rotation="100 MB",
            retention="1 week",
            level=log_level,
        )

        self.models: Dict[str, ModelMetadata] = {}
        self.gpu_manager = GPUManager()
        self.allocation_strategy = allocation_strategy
        self.memory_buffer = memory_buffer
        self.max_cpu_models = max_cpu_models
        self.use_multiprocessing = use_multiprocessing

        # Initialize locks and queues for multiprocessing
        self.manager = (
            multiprocessing.Manager() if use_multiprocessing else None
        )
        self.task_queues: Dict[str, Any] = (
            self.manager.dict() if use_multiprocessing else {}
        )
        self.result_queues: Dict[str, Any] = (
            self.manager.dict() if use_multiprocessing else {}
        )
        self.model_locks: Dict[str, Any] = {}

        logger.info(
            f"ModelGrid initialized with {len(self.gpu_manager.gpus)} GPUs"
        )
        logger.info(
            f"Using allocation strategy: {allocation_strategy.value}"
        )

    def add_model(
        self,
        model_name: str,
        model: Any,
        model_type: Optional[ModelType] = None,
        memory_override: Optional[float] = None,
    ) -> bool:
        """
        Add a model to the manager.

        Args:
            model_name: Unique name for the model
            model: The model object or path
            model_type: Type of the model (will be auto-detected if not provided)
            memory_override: Override the automatic memory calculation (in GB)

        Returns:
            Success status
        """
        if model_name in self.models:
            logger.warning(
                f"Model '{model_name}' already exists. Use update_model to replace it."
            )
            return False

        # Auto-detect model type if not provided
        if model_type is None:
            if isinstance(model, str):
                if os.path.exists(model) and TRANSFORMERS_AVAILABLE:
                    model_type = ModelType.HUGGINGFACE
                else:
                    model_type = ModelType.UNKNOWN
            elif isinstance(model, torch.nn.Module):
                model_type = ModelType.PYTORCH
            elif TRANSFORMERS_AVAILABLE and isinstance(
                model, transformers.PreTrainedModel
            ):
                model_type = ModelType.HUGGINGFACE
            else:
                model_type = ModelType.UNKNOWN

        # Calculate memory requirements
        if memory_override is not None:
            memory_required = memory_override
        else:
            if model_type == ModelType.PYTORCH:
                memory_required = (
                    ModelMemoryCalculator.get_pytorch_model_size(
                        model
                    )
                )
            elif model_type == ModelType.HUGGINGFACE:
                memory_required = (
                    ModelMemoryCalculator.get_huggingface_model_size(
                        model
                    )
                )
            else:
                logger.warning(
                    f"Unknown model type for '{model_name}'. Using default memory estimation."
                )
                memory_required = 2.0  # Default estimation

        # Create model metadata
        model_metadata = ModelMetadata(
            name=model_name,
            model_type=model_type,
            memory_required=memory_required,
            model=model,
            loaded=False,
        )

        self.models[model_name] = model_metadata
        logger.info(
            f"Added model '{model_name}' ({model_type.value}) with {memory_required:.2f} GB memory requirement"
        )

        # Initialize multiprocessing resources for this model
        if self.use_multiprocessing:
            self.task_queues[model_name] = self.manager.Queue()
            self.result_queues[model_name] = self.manager.Queue()
            self.model_locks[model_name] = self.manager.Lock()

        return True

    def remove_model(self, model_name: str) -> bool:
        """
        Remove a model from the manager.

        Args:
            model_name: Name of the model to remove

        Returns:
            Success status
        """
        if model_name not in self.models:
            logger.warning(f"Model '{model_name}' does not exist.")
            return False

        model_metadata = self.models[model_name]

        # Terminate the model process if running
        if (
            model_metadata.process is not None
            and model_metadata.process.is_alive()
        ):
            logger.info(
                f"Terminating process for model '{model_name}'"
            )
            model_metadata.process.terminate()
            model_metadata.process.join(timeout=5)
            if model_metadata.process.is_alive():
                logger.warning(
                    f"Process for model '{model_name}' did not terminate gracefully. Killing..."
                )
                model_metadata.process.kill()

        # Remove from GPU if loaded
        if (
            model_metadata.loaded
            and model_metadata.device is not None
        ):
            gpu_id = model_metadata.device.index
            for gpu in self.gpu_manager.gpus:
                if gpu.id == gpu_id and model_name in gpu.models:
                    gpu.models.remove(model_name)
                    logger.info(
                        f"Removed model '{model_name}' from GPU {gpu_id}"
                    )

            # Update GPU memory info
            self.gpu_manager.update_gpu_memory_info()

        # Clean up multiprocessing resources
        if self.use_multiprocessing:
            if model_name in self.task_queues:
                del self.task_queues[model_name]
            if model_name in self.result_queues:
                del self.result_queues[model_name]
            if model_name in self.model_locks:
                del self.model_locks[model_name]

        # Remove model metadata
        del self.models[model_name]

        logger.info(f"Removed model '{model_name}'")
        return True

    def _find_best_gpu_for_model(
        self, model_metadata: ModelMetadata
    ) -> Optional[GPUMetadata]:
        """
        Find the best GPU for a given model based on the allocation strategy.

        Args:
            model_metadata: Metadata for the model

        Returns:
            Best GPU metadata or None if no suitable GPU found
        """
        model_memory = (
            model_metadata.memory_required + self.memory_buffer
        )

        # Update GPU memory info before allocation
        self.gpu_manager.update_gpu_memory_info()

        # Find available GPUs that can fit the model
        available_gpus = [
            gpu
            for gpu in self.gpu_manager.gpus
            if gpu.available_memory >= model_memory
        ]

        if not available_gpus:
            logger.warning(
                f"No GPU with sufficient memory for model '{model_metadata.name}' "
                f"(requires {model_memory:.2f} GB)"
            )
            return None

        # Apply allocation strategy
        if self.allocation_strategy == GPUAllocationStrategy.FILL_GPU:
            # Sort by number of models (ascending) and then by available memory (descending)
            return sorted(
                available_gpus,
                key=lambda g: (len(g.models), -g.available_memory),
            )[0]

        elif (
            self.allocation_strategy
            == GPUAllocationStrategy.DISTRIBUTE
        ):
            # Sort by number of models (ascending)
            return sorted(
                available_gpus, key=lambda g: len(g.models)
            )[0]

        elif (
            self.allocation_strategy
            == GPUAllocationStrategy.MEMORY_OPTIMIZED
        ):
            # Sort by available memory (ascending) but ensure it fits
            return sorted(
                available_gpus, key=lambda g: g.available_memory
            )[0]

        # Default fallback
        return available_gpus[0]

    def allocate_all_models(self) -> Dict[str, Optional[int]]:
        """
        Allocate all models to GPUs based on the allocation strategy.

        Returns:
            Dict mapping model names to allocated GPU IDs (or None if on CPU)
        """
        # Sort models by memory requirement (descending)
        sorted_models = sorted(
            self.models.values(),
            key=lambda m: m.memory_required,
            reverse=True,
        )

        allocations = {}

        for model_metadata in sorted_models:
            best_gpu = self._find_best_gpu_for_model(model_metadata)

            if best_gpu is not None:
                # Allocate model to GPU
                gpu_id = best_gpu.id
                model_metadata.device = best_gpu.device
                best_gpu.models.append(model_metadata.name)
                best_gpu.available_memory -= (
                    model_metadata.memory_required
                    + self.memory_buffer
                )

                allocations[model_metadata.name] = gpu_id
                logger.info(
                    f"Allocated model '{model_metadata.name}' to GPU {gpu_id} "
                    f"({best_gpu.available_memory:.2f} GB remaining)"
                )
            else:
                # No suitable GPU found, keep model on CPU if allowed
                if (
                    len(
                        [m for m in allocations.values() if m is None]
                    )
                    < self.max_cpu_models
                ):
                    model_metadata.device = None
                    allocations[model_metadata.name] = None
                    logger.info(
                        f"Keeping model '{model_metadata.name}' on CPU (no suitable GPU)"
                    )
                else:
                    logger.warning(
                        f"Cannot allocate model '{model_metadata.name}'. "
                        f"No GPU space available and max_cpu_models limit reached."
                    )

        return allocations

    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model to its allocated device.

        Args:
            model_name: Name of the model to load

        Returns:
            Success status
        """
        if model_name not in self.models:
            logger.warning(f"Model '{model_name}' does not exist.")
            return False

        model_metadata = self.models[model_name]

        # Skip if already loaded
        if model_metadata.loaded:
            logger.info(f"Model '{model_name}' is already loaded.")
            return True

        # Allocate to GPU if not already allocated
        if model_metadata.device is None:
            best_gpu = self._find_best_gpu_for_model(model_metadata)
            if best_gpu is not None:
                model_metadata.device = best_gpu.device
                best_gpu.models.append(model_name)
                best_gpu.available_memory -= (
                    model_metadata.memory_required
                    + self.memory_buffer
                )
                logger.info(
                    f"Allocated model '{model_name}' to GPU {best_gpu.id} "
                    f"({best_gpu.available_memory:.2f} GB remaining)"
                )

        try:
            device_str = (
                "cpu"
                if model_metadata.device is None
                else str(model_metadata.device)
            )
            logger.info(
                f"Loading model '{model_name}' to {device_str}"
            )

            # Load based on model type
            if model_metadata.model_type == ModelType.PYTORCH:
                if isinstance(model_metadata.model, torch.nn.Module):
                    model_metadata.model.to(
                        model_metadata.device or "cpu"
                    )
                else:
                    logger.error(
                        f"Model '{model_name}' is not a valid PyTorch module."
                    )
                    return False

            elif model_metadata.model_type == ModelType.HUGGINGFACE:
                if TRANSFORMERS_AVAILABLE:
                    if isinstance(model_metadata.model, str):
                        # Load from path
                        logger.info(
                            f"Loading HuggingFace model from {model_metadata.model}"
                        )
                        loaded_model = AutoModel.from_pretrained(
                            model_metadata.model
                        )
                        loaded_model.to(
                            model_metadata.device or "cpu"
                        )
                        model_metadata.model = loaded_model
                    elif isinstance(
                        model_metadata.model,
                        transformers.PreTrainedModel,
                    ):
                        # Move existing model to device
                        model_metadata.model.to(
                            model_metadata.device or "cpu"
                        )
                    else:
                        logger.error(
                            f"Model '{model_name}' is not a valid HuggingFace model."
                        )
                        return False
                else:
                    logger.error(
                        "Transformers package not available. Cannot load HuggingFace model."
                    )
                    return False
            else:
                logger.error(
                    f"Unknown model type for '{model_name}'."
                )
                return False

            model_metadata.loaded = True

            # Start model process if using multiprocessing
            if self.use_multiprocessing:
                self._start_model_process(model_name)

            logger.info(f"Successfully loaded model '{model_name}'")
            return True

        except Exception as e:
            logger.error(
                f"Error loading model '{model_name}': {str(e)}"
            )
            # Try to clean up GPU allocation if failed
            if model_metadata.device is not None:
                gpu_id = model_metadata.device.index
                for gpu in self.gpu_manager.gpus:
                    if gpu.id == gpu_id and model_name in gpu.models:
                        gpu.models.remove(model_name)
                        gpu.available_memory += (
                            model_metadata.memory_required
                            + self.memory_buffer
                        )
                model_metadata.device = None

            self.gpu_manager.update_gpu_memory_info()
            return False

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model from its device.

        Args:
            model_name: Name of the model to unload

        Returns:
            Success status
        """
        if model_name not in self.models:
            logger.warning(f"Model '{model_name}' does not exist.")
            return False

        model_metadata = self.models[model_name]

        # Skip if not loaded
        if not model_metadata.loaded:
            logger.info(f"Model '{model_name}' is not loaded.")
            return True

        try:
            # Stop model process if using multiprocessing
            if (
                self.use_multiprocessing
                and model_metadata.process is not None
            ):
                logger.info(
                    f"Stopping process for model '{model_name}'"
                )
                model_metadata.process.terminate()
                model_metadata.process.join(timeout=5)
                if model_metadata.process.is_alive():
                    logger.warning(
                        f"Process for model '{model_name}' did not terminate gracefully. Killing..."
                    )
                    model_metadata.process.kill()
                model_metadata.process = None

            # Move model to CPU and clean up
            if (
                model_metadata.device is not None
                and model_metadata.device.type == "cuda"
            ):
                logger.info(
                    f"Unloading model '{model_name}' from {model_metadata.device}"
                )

                # Update GPU allocation
                gpu_id = model_metadata.device.index
                for gpu in self.gpu_manager.gpus:
                    if gpu.id == gpu_id and model_name in gpu.models:
                        gpu.models.remove(model_name)
                        gpu.available_memory += (
                            model_metadata.memory_required
                            + self.memory_buffer
                        )

                # Move model to CPU if it's a PyTorch module
                if isinstance(model_metadata.model, torch.nn.Module):
                    model_metadata.model.to("cpu")

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            model_metadata.device = None
            model_metadata.loaded = False

            # Update GPU memory info
            self.gpu_manager.update_gpu_memory_info()

            logger.info(f"Successfully unloaded model '{model_name}'")
            return True

        except Exception as e:
            logger.error(
                f"Error unloading model '{model_name}': {str(e)}"
            )
            return False

    def load_all_models(self) -> Dict[str, bool]:
        """
        Load all models to their allocated devices.

        Returns:
            Dict mapping model names to load success status
        """
        # First allocate all models
        self.allocate_all_models()

        # Then load each model
        results = {}
        for model_name in self.models:
            results[model_name] = self.load_model(model_name)

        return results

    def unload_all_models(self) -> Dict[str, bool]:
        """
        Unload all models from their devices.

        Returns:
            Dict mapping model names to unload success status
        """
        results = {}
        for model_name in self.models:
            results[model_name] = self.unload_model(model_name)

        return results

    def _start_model_process(self, model_name: str) -> bool:
        """
        Start a dedicated process for a model.

        Args:
            model_name: Name of the model

        Returns:
            Success status
        """
        if not self.use_multiprocessing:
            logger.warning(
                "Multiprocessing is disabled. Cannot start model process."
            )
            return False

        if model_name not in self.models:
            logger.warning(f"Model '{model_name}' does not exist.")
            return False

        model_metadata = self.models[model_name]

        if (
            model_metadata.process is not None
            and model_metadata.process.is_alive()
        ):
            logger.info(
                f"Process for model '{model_name}' is already running."
            )
            return True

        try:
            # Create a new process for the model
            process = multiprocessing.Process(
                target=self._model_process_worker,
                args=(
                    model_name,
                    model_metadata.model_type,
                    self.task_queues[model_name],
                    self.result_queues[model_name],
                    (
                        model_metadata.device.index
                        if model_metadata.device is not None
                        else None
                    ),
                ),
                daemon=True,
            )

            process.start()
            model_metadata.process = process

            logger.info(
                f"Started process for model '{model_name}' (PID: {process.pid})"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error starting process for model '{model_name}': {str(e)}"
            )
            return False

    def _model_process_worker(
        self,
        model_name: str,
        model_type: ModelType,
        task_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
        gpu_id: Optional[int],
    ) -> None:
        """
        Worker function for model processes.

        Args:
            model_name: Name of the model
            model_type: Type of the model
            task_queue: Queue for receiving tasks
            result_queue: Queue for sending results
            gpu_id: GPU device ID or None for CPU
        """
        try:
            # Configure device
            if gpu_id is not None:
                device = torch.device(f"cuda:{gpu_id}")
                torch.cuda.set_device(device)
            else:
                device = torch.device("cpu")

            logger.info(
                f"Model process for '{model_name}' started on {device}"
            )

            # Process tasks from the queue
            while True:
                try:
                    # Get task from queue with timeout
                    task_id, task_type, task_data = task_queue.get(
                        timeout=1.0
                    )

                    logger.debug(
                        f"Model '{model_name}' processing task {task_id}: {task_type}"
                    )

                    # Process task based on task_type
                    try:
                        if task_type == "run_model":
                            # Run the model on the task data
                            # This would be implemented based on the specific model type
                            result = {
                                "status": "success",
                                "result": "Model output placeholder",
                            }
                        else:
                            result = {
                                "status": "error",
                                "error": f"Unknown task type: {task_type}",
                            }
                    except Exception as e:
                        logger.error(
                            f"Error processing task {task_id} for model '{model_name}': {str(e)}"
                        )
                        result = {"status": "error", "error": str(e)}

                    # Send result back
                    result_queue.put((task_id, result))
                    logger.debug(
                        f"Model '{model_name}' completed task {task_id}"
                    )

                except queue.Empty:
                    # No tasks in queue, just continue
                    continue

        except KeyboardInterrupt:
            logger.info(
                f"Model process for '{model_name}' interrupted"
            )
        except Exception as e:
            logger.error(
                f"Error in model process for '{model_name}': {str(e)}"
            )
        finally:
            logger.info(f"Model process for '{model_name}' exiting")

    @contextmanager
    def _model_lock(self, model_name: str) -> None:
        """
        Context manager for acquiring model lock.

        Args:
            model_name: Name of the model
        """
        if (
            not self.use_multiprocessing
            or model_name not in self.model_locks
        ):
            # No-op if not using multiprocessing
            yield
            return

        lock = self.model_locks[model_name]
        try:
            lock.acquire()
            yield
        finally:
            lock.release()

    def run(
        self,
        task: Union[str, List[str]],
        model_names: Optional[List[str]] = None,
        input_data: Any = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Run a task on specific models or all models.

        Args:
            task: Task name or list of task names to run
            model_names: List of model names to run the task on (None for all loaded models)
            input_data: Input data for the task
            timeout: Timeout in seconds

        Returns:
            Dict mapping model names to results
        """
        # Normalize task to list
        if isinstance(task, str):
            tasks = [task]
        else:
            tasks = task

        # Determine which models to run on
        if model_names is None:
            target_models = [
                name
                for name, meta in self.models.items()
                if meta.loaded
            ]
        else:
            target_models = [
                name
                for name in model_names
                if name in self.models and self.models[name].loaded
            ]

        if not target_models:
            logger.warning(
                "No loaded models available for running tasks."
            )
            return {}

        logger.info(
            f"Running tasks {tasks} on models: {', '.join(target_models)}"
        )

        results = {}

        # Run tasks on each model
        for model_name in target_models:
            model_metadata = self.models[model_name]

            try:
                if (
                    self.use_multiprocessing
                    and model_metadata.process is not None
                ):
                    # Run in separate process
                    results[model_name] = self._run_in_process(
                        model_name, tasks, input_data, timeout
                    )
                else:
                    # Run in current process
                    results[model_name] = (
                        self._run_in_current_process(
                            model_name, tasks, input_data
                        )
                    )

            except Exception as e:
                logger.error(
                    f"Error running tasks on model '{model_name}': {str(e)}"
                )
                results[model_name] = {
                    "status": "error",
                    "error": str(e),
                }

        return results

    def _run_in_process(
        self,
        model_name: str,
        tasks: List[str],
        input_data: Any,
        timeout: float,
    ) -> Dict[str, Any]:
        """
        Run tasks on a model in a separate process.

        Args:
            model_name: Name of the model
            tasks: List of tasks to run
            input_data: Input data for the tasks
            timeout: Timeout in seconds

        Returns:
            Task results
        """
        task_id = str(uuid.uuid4())
        task_queue = self.task_queues[model_name]
        result_queue = self.result_queues[model_name]

        # Send task to model process
        task_queue.put((task_id, tasks[0], input_data))

        # Wait for result
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if result is available
                result_task_id, result = result_queue.get(block=False)

                if result_task_id == task_id:
                    return result
                else:
                    # Put back other task results
                    result_queue.put((result_task_id, result))

            except queue.Empty:
                # No results yet, wait a bit
                time.sleep(0.1)

        # Timeout
        logger.warning(
            f"Timeout waiting for tasks on model '{model_name}'"
        )
        return {"status": "error", "error": "Timeout"}

    def _run_in_current_process(
        self, model_name: str, tasks: List[str], input_data: Any
    ) -> Dict[str, Any]:
        """
        Run tasks on a model in the current process.

        Args:
            model_name: Name of the model
            tasks: List of tasks to run
            input_data: Input data for the tasks

        Returns:
            Task results
        """
        model_metadata = self.models[model_name]

        with self._model_lock(model_name):
            try:
                # This would need to be implemented based on the specific model types
                # and tasks supported. Here's a simple placeholder:

                if model_metadata.model_type == ModelType.PYTORCH:
                    # Run PyTorch model
                    return {
                        "status": "success",
                        "result": "PyTorch model output placeholder",
                    }

                elif (
                    model_metadata.model_type == ModelType.HUGGINGFACE
                ):
                    # Run Hugging Face model
                    return {
                        "status": "success",
                        "result": "Hugging Face model output placeholder",
                    }

                else:
                    return {
                        "status": "error",
                        "error": f"Unsupported model type: {model_metadata.model_type}",
                    }

            except Exception as e:
                logger.error(
                    f"Error running tasks on model '{model_name}': {str(e)}"
                )
                return {"status": "error", "error": str(e)}

    def get_gpu_status(self) -> List[Dict[str, Any]]:
        """
        Get status information for all GPUs.

        Returns:
            List of GPU status dictionaries
        """
        # Update GPU memory info
        self.gpu_manager.update_gpu_memory_info()

        gpu_status = []
        for gpu in self.gpu_manager.gpus:
            status = {
                "id": gpu.id,
                "total_memory": gpu.total_memory,
                "available_memory": gpu.available_memory,
                "used_memory": gpu.total_memory
                - gpu.available_memory,
                "utilization": (
                    gpu.total_memory - gpu.available_memory
                )
                / gpu.total_memory,
                "models": gpu.models,
            }
            gpu_status.append(status)

        return gpu_status

    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all models.

        Returns:
            Dict mapping model names to status dictionaries
        """
        model_status = {}
        for name, metadata in self.models.items():
            status = {
                "name": name,
                "type": metadata.model_type.value,
                "memory_required": metadata.memory_required,
                "loaded": metadata.loaded,
                "device": (
                    str(metadata.device)
                    if metadata.device is not None
                    else "cpu"
                ),
                "process_running": metadata.process is not None
                and metadata.process.is_alive(),
            }
            model_status[name] = status

        return model_status


class ModelWithCustomRunMethod:
    """
    Base class for models with custom run methods.

    Extend this class to implement custom run methods for specific model types.
    """

    def __init__(
        self, model: Any, device: Optional[torch.device] = None
    ):
        """
        Initialize the model wrapper.

        Args:
            model: The model object
            device: Device to run the model on
        """
        self.model = model
        self.device = device

    def run(self, task: str, input_data: Any) -> Any:
        """
        Run a task on the model.

        Args:
            task: Task name
            input_data: Input data for the task

        Returns:
            Task result
        """
        raise NotImplementedError(
            "Subclasses must implement this method"
        )


class PyTorchModelWrapper(ModelWithCustomRunMethod):
    """
    Wrapper for PyTorch models with custom run methods.
    """

    def run(self, task: str, input_data: Any) -> Any:
        """
        Run a task on a PyTorch model.

        Args:
            task: Task name
            input_data: Input data for the task

        Returns:
            Task result
        """
        # Example implementation for common PyTorch tasks
        if task == "forward":
            # Ensure model is in eval mode
            self.model.eval()

            # Convert input to tensor if needed
            if not isinstance(input_data, torch.Tensor):
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.from_numpy(input_data).to(
                        self.device
                    )
                else:
                    input_tensor = torch.tensor(input_data).to(
                        self.device
                    )
            else:
                input_tensor = input_data.to(self.device)

            # Run forward pass
            with torch.no_grad():
                output = self.model(input_tensor)

            # Convert output to numpy if needed
            if isinstance(output, torch.Tensor):
                return output.cpu().numpy()
            else:
                return output

        elif task == "predict":
            # Similar to forward but with different post-processing
            self.model.eval()

            # Convert input to tensor if needed
            if not isinstance(input_data, torch.Tensor):
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.from_numpy(input_data).to(
                        self.device
                    )
                else:
                    input_tensor = torch.tensor(input_data).to(
                        self.device
                    )
            else:
                input_tensor = input_data.to(self.device)

            # Run prediction
            with torch.no_grad():
                output = self.model(input_tensor)

                # Apply softmax if output is logits
                if len(output.shape) > 1 and output.shape[1] > 1:
                    probs = torch.nn.functional.softmax(output, dim=1)
                    predicted_class = torch.argmax(probs, dim=1)
                    return {
                        "probabilities": probs.cpu().numpy(),
                        "predicted_class": predicted_class.cpu().numpy(),
                    }
                else:
                    return output.cpu().numpy()
        else:
            raise ValueError(f"Unsupported task: {task}")


class HuggingFaceModelWrapper(ModelWithCustomRunMethod):
    """
    Wrapper for Hugging Face models with custom run methods.
    """

    def run(self, task: str, input_data: Any) -> Any:
        """
        Run a task on a Hugging Face model.

        Args:
            task: Task name
            input_data: Input data for the task

        Returns:
            Task result
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers package not available.")

        # Example implementation for common Hugging Face tasks
        if task == "generate":
            # Generate text
            return self.model.generate(**input_data)

        elif task == "encode":
            # Encode text
            return self.model.encode(input_data)

        elif task == "predict":
            # Make predictions
            return self.model(**input_data)

        else:
            raise ValueError(f"Unsupported task: {task}")


# # Example usage
# if __name__ == "__main__":
#     # Initialize model manager
#     manager = ModelGrid(
#         allocation_strategy=GPUAllocationStrategy.MEMORY_OPTIMIZED,
#         memory_buffer=0.5,
#         max_cpu_models=1,
#         use_multiprocessing=True,
#         log_level="INFO",
#     )

#     # # Add models
#     model1 = torch.nn.Sequential(
#         torch.nn.Linear(10, 10),
#         torch.nn.ReLU(),
#         torch.nn.Linear(10, 2),
#     )
#     manager.add_model("small_model", model1, ModelType.PYTORCH)

#     # Add more models if available
#     if TRANSFORMERS_AVAILABLE:
#         manager.add_model(
#             "bert_model", "bert-base-uncased", ModelType.HUGGINGFACE
#         )

#     # Allocate and load models
#     manager.load_all_models()

#     # Print GPU status
#     print("GPU Status:")
#     for gpu in manager.get_gpu_status():
#         print(
#             f"GPU {gpu['id']}: {gpu['available_memory']:.2f} GB / {gpu['total_memory']:.2f} GB"
#         )
#         print(f"  Models: {', '.join(gpu['models'])}")

#     # Run a task on all models
#     results = manager.run("forward", input_data=torch.randn(1, 10))

#     # Unload all models
#     manager.unload_all_models()
