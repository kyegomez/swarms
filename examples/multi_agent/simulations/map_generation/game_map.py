"""
Production-grade AI Vision Pipeline for depth estimation, segmentation, object detection,
and 3D point cloud generation.

This module provides a comprehensive pipeline that combines MiDaS for depth estimation,
SAM (Segment Anything Model) for semantic segmentation, YOLOv8 for object detection,
and Open3D for 3D point cloud generation.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import open3d as o3d
from loguru import logger


warnings.filterwarnings("ignore")


# Third-party model imports
try:
    import timm
    from segment_anything import (
        SamAutomaticMaskGenerator,
        sam_model_registry,
    )
    from ultralytics import YOLO
except ImportError as e:
    logger.error(f"Missing required dependencies: {e}")
    sys.exit(1)


class AIVisionPipeline:
    """
    A comprehensive AI vision pipeline that performs depth estimation, semantic segmentation,
    object detection, and 3D point cloud generation from input images.

    This class integrates multiple state-of-the-art models:
    - MiDaS for monocular depth estimation
    - SAM (Segment Anything Model) for semantic segmentation
    - YOLOv8 for object detection
    - Open3D for 3D point cloud generation

    Attributes:
        model_dir (Path): Directory where models are stored
        device (torch.device): Computing device (CPU/CUDA)
        midas_model: Loaded MiDaS depth estimation model
        midas_transform: MiDaS preprocessing transforms
        sam_generator: SAM automatic mask generator
        yolo_model: YOLOv8 object detection model

    Example:
        >>> pipeline = AIVisionPipeline()
        >>> results = pipeline.process_image("path/to/image.jpg")
        >>> point_cloud = results["point_cloud"]
    """

    def __init__(
        self,
        model_dir: str = "./models",
        device: Optional[str] = None,
        midas_model_type: str = "MiDaS",
        sam_model_type: str = "vit_b",
        yolo_model_path: str = "yolov8n.pt",
        log_level: str = "INFO",
    ) -> None:
        """
        Initialize the AI Vision Pipeline.

        Args:
            model_dir: Directory to store downloaded models
            device: Computing device ('cpu', 'cuda', or None for auto-detection)
            midas_model_type: MiDaS model variant ('MiDaS', 'MiDaS_small', 'DPT_Large', etc.)
            sam_model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            yolo_model_path: Path to YOLOv8 model weights
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

        Raises:
            RuntimeError: If required models cannot be loaded
            FileNotFoundError: If model files are not found
        """
        # Setup logging
        logger.remove()
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )

        # Initialize attributes
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Model configuration
        self.midas_model_type = midas_model_type
        self.sam_model_type = sam_model_type
        self.yolo_model_path = yolo_model_path

        # Initialize model placeholders
        self.midas_model: Optional[torch.nn.Module] = None
        self.midas_transform: Optional[transforms.Compose] = None
        self.sam_generator: Optional[SamAutomaticMaskGenerator] = None
        self.yolo_model: Optional[YOLO] = None

        # Load all models
        self._setup_models()

        logger.success("AI Vision Pipeline initialized successfully")

    def _setup_models(self) -> None:
        """
        Load and initialize all AI models with proper error handling.

        Raises:
            RuntimeError: If any model fails to load
        """
        try:
            self._load_midas_model()
            self._load_sam_model()
            self._load_yolo_model()
        except Exception as e:
            logger.error(f"Failed to setup models: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def _load_midas_model(self) -> None:
        """Load MiDaS depth estimation model."""
        try:
            logger.info(
                f"Loading MiDaS model: {self.midas_model_type}"
            )

            # Load MiDaS model from torch hub
            self.midas_model = torch.hub.load(
                "intel-isl/MiDaS",
                self.midas_model_type,
                pretrained=True,
            )
            self.midas_model.to(self.device)
            self.midas_model.eval()

            # Load corresponding transforms
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms"
            )

            if self.midas_model_type in ["DPT_Large", "DPT_Hybrid"]:
                self.midas_transform = midas_transforms.dpt_transform
            else:
                self.midas_transform = (
                    midas_transforms.default_transform
                )

            logger.success("MiDaS model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            raise

    def _load_sam_model(self) -> None:
        """Load SAM (Segment Anything Model) for semantic segmentation."""
        try:
            logger.info(f"Loading SAM model: {self.sam_model_type}")

            # SAM model checkpoints mapping
            sam_checkpoint_urls = {
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            }

            checkpoint_path = (
                self.model_dir / f"sam_{self.sam_model_type}.pth"
            )

            # Download checkpoint if not exists
            if not checkpoint_path.exists():
                logger.info(
                    f"Downloading SAM checkpoint to {checkpoint_path}"
                )
                import urllib.request

                urllib.request.urlretrieve(
                    sam_checkpoint_urls[self.sam_model_type],
                    checkpoint_path,
                )

            # Load SAM model
            sam = sam_model_registry[self.sam_model_type](
                checkpoint=str(checkpoint_path)
            )
            sam.to(self.device)

            # Create automatic mask generator
            self.sam_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )

            logger.success("SAM model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            raise

    def _load_yolo_model(self) -> None:
        """Load YOLOv8 object detection model."""
        try:
            logger.info(
                f"Loading YOLOv8 model: {self.yolo_model_path}"
            )

            self.yolo_model = YOLO(self.yolo_model_path)

            # Move to appropriate device
            if self.device.type == "cuda":
                self.yolo_model.to(self.device)

            logger.success("YOLOv8 model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise

    def _load_and_preprocess_image(
        self, image_path: Union[str, Path]
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Load and preprocess input image.

        Args:
            image_path: Path to the input image (JPG or PNG)

        Returns:
            Tuple of (opencv_image, pil_image)

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is not supported
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            raise ValueError(
                f"Unsupported image format: {image_path.suffix}"
            )

        try:
            # Load with OpenCV (BGR format)
            cv_image = cv2.imread(str(image_path))
            if cv_image is None:
                raise ValueError(
                    f"Could not load image: {image_path}"
                )

            # Convert BGR to RGB for PIL
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            logger.debug(
                f"Loaded image: {image_path} ({rgb_image.shape})"
            )
            return rgb_image, pil_image

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map using MiDaS model.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format

        Returns:
            Depth map as numpy array (H, W)

        Raises:
            RuntimeError: If depth estimation fails
        """
        try:
            logger.debug("Estimating depth with MiDaS")

            # Preprocess image for MiDaS
            input_tensor = self.midas_transform(image).to(self.device)

            # Perform inference
            with torch.no_grad():
                depth_map = self.midas_model(input_tensor)
                depth_map = torch.nn.functional.interpolate(
                    depth_map.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Convert to numpy
            depth_numpy = depth_map.cpu().numpy()

            # Normalize depth values
            depth_numpy = (depth_numpy - depth_numpy.min()) / (
                depth_numpy.max() - depth_numpy.min()
            )

            logger.debug(
                f"Depth estimation completed. Shape: {depth_numpy.shape}"
            )
            return depth_numpy

        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            raise RuntimeError(f"Depth estimation error: {e}")

    def segment_image(
        self, image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic segmentation using SAM.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format

        Returns:
            List of segmentation masks with metadata

        Raises:
            RuntimeError: If segmentation fails
        """
        try:
            logger.debug("Performing segmentation with SAM")

            # Generate masks
            masks = self.sam_generator.generate(image)

            logger.debug(f"Generated {len(masks)} segmentation masks")
            return masks

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise RuntimeError(f"Segmentation error: {e}")

    def detect_objects(
        self, image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Perform object detection using YOLOv8.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format

        Returns:
            List of detected objects with bounding boxes and confidence scores

        Raises:
            RuntimeError: If object detection fails
        """
        try:
            logger.debug("Performing object detection with YOLOv8")

            # Run inference
            results = self.yolo_model(image, verbose=False)

            # Extract detections
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        detection = {
                            "bbox": boxes.xyxy[i]
                            .cpu()
                            .numpy(),  # [x1, y1, x2, y2]
                            "confidence": float(
                                boxes.conf[i].cpu().numpy()
                            ),
                            "class_id": int(
                                boxes.cls[i].cpu().numpy()
                            ),
                            "class_name": result.names[
                                int(boxes.cls[i].cpu().numpy())
                            ],
                        }
                        detections.append(detection)

            logger.debug(f"Detected {len(detections)} objects")
            return detections

        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            raise RuntimeError(f"Object detection error: {e}")

    def generate_point_cloud(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        masks: Optional[List[Dict[str, Any]]] = None,
    ) -> o3d.geometry.PointCloud:
        """
        Generate 3D point cloud from image and depth data.

        Args:
            image: RGB image array (H, W, 3)
            depth_map: Depth map array (H, W)
            masks: Optional segmentation masks for point cloud filtering

        Returns:
            Open3D PointCloud object

        Raises:
            ValueError: If input dimensions don't match
            RuntimeError: If point cloud generation fails
        """
        try:
            logger.debug("Generating 3D point cloud")

            if image.shape[:2] != depth_map.shape:
                raise ValueError(
                    "Image and depth map dimensions must match"
                )

            height, width = depth_map.shape

            # Create intrinsic camera parameters (assuming standard camera)
            fx = fy = width  # Focal length approximation
            cx, cy = (
                width / 2,
                height / 2,
            )  # Principal point at image center

            # Create coordinate grids
            u, v = np.meshgrid(np.arange(width), np.arange(height))

            # Convert depth to actual distances (inverse depth)
            # MiDaS outputs inverse depth, so we invert it
            z = 1.0 / (
                depth_map + 1e-6
            )  # Add small epsilon to avoid division by zero

            # Back-project to 3D coordinates
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            # Create point cloud
            points = np.stack(
                [x.flatten(), y.flatten(), z.flatten()], axis=1
            )
            colors = (
                image.reshape(-1, 3) / 255.0
            )  # Normalize colors to [0, 1]

            # Filter out invalid points
            valid_mask = np.isfinite(points).all(axis=1) & (
                z.flatten() > 0
            )
            points = points[valid_mask]
            colors = colors[valid_mask]

            # Create Open3D point cloud
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

            # Optional: Filter by segmentation masks
            if masks and len(masks) > 0:
                # Use the largest mask for filtering
                largest_mask = max(masks, key=lambda x: x["area"])
                mask_2d = largest_mask["segmentation"]
                mask_1d = mask_2d.flatten()[valid_mask]

                filtered_points = points[mask_1d]
                filtered_colors = colors[mask_1d]

                point_cloud.points = o3d.utility.Vector3dVector(
                    filtered_points
                )
                point_cloud.colors = o3d.utility.Vector3dVector(
                    filtered_colors
                )

            # Remove statistical outliers
            point_cloud, _ = point_cloud.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )

            logger.debug(
                f"Generated point cloud with {len(point_cloud.points)} points"
            )
            return point_cloud

        except Exception as e:
            logger.error(f"Point cloud generation failed: {e}")
            raise RuntimeError(f"Point cloud generation error: {e}")

    def process_image(
        self, image_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Process a single image through the complete AI vision pipeline.

        Args:
            image_path: Path to input image (JPG or PNG)

        Returns:
            Dictionary containing all processing results:
            - 'image': Original RGB image
            - 'depth_map': Depth estimation result
            - 'segmentation_masks': SAM segmentation results
            - 'detections': YOLO object detection results
            - 'point_cloud': Open3D point cloud object

        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If any processing step fails
        """
        try:
            logger.info(f"Processing image: {image_path}")

            # Load and preprocess image
            rgb_image, pil_image = self._load_and_preprocess_image(
                image_path
            )

            # Depth estimation
            depth_map = self.estimate_depth(rgb_image)

            # Semantic segmentation
            segmentation_masks = self.segment_image(rgb_image)

            # Object detection
            detections = self.detect_objects(rgb_image)

            # 3D point cloud generation
            point_cloud = self.generate_point_cloud(
                rgb_image, depth_map, segmentation_masks
            )

            # Compile results
            results = {
                "image": rgb_image,
                "depth_map": depth_map,
                "segmentation_masks": segmentation_masks,
                "detections": detections,
                "point_cloud": point_cloud,
                "metadata": {
                    "image_shape": rgb_image.shape,
                    "num_segments": len(segmentation_masks),
                    "num_detections": len(detections),
                    "num_points": len(point_cloud.points),
                },
            }

            logger.success("Image processing completed successfully")
            logger.info(f"Results: {results['metadata']}")

            return results

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise

    def save_point_cloud(
        self,
        point_cloud: o3d.geometry.PointCloud,
        output_path: Union[str, Path],
    ) -> None:
        """
        Save point cloud to file.

        Args:
            point_cloud: Open3D PointCloud object
            output_path: Output file path (.ply, .pcd, .xyz)

        Raises:
            RuntimeError: If saving fails
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            success = o3d.io.write_point_cloud(
                str(output_path), point_cloud
            )

            if not success:
                raise RuntimeError("Failed to write point cloud file")

            logger.success(f"Point cloud saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save point cloud: {e}")
            raise RuntimeError(f"Point cloud save error: {e}")

    def visualize_point_cloud(
        self, point_cloud: o3d.geometry.PointCloud
    ) -> None:
        """
        Visualize point cloud using Open3D viewer.

        Args:
            point_cloud: Open3D PointCloud object to visualize
        """
        try:
            logger.info("Opening point cloud visualization")
            o3d.visualization.draw_geometries([point_cloud])
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    try:
        # Initialize pipeline
        pipeline = AIVisionPipeline(
            model_dir="./models", log_level="INFO"
        )

        # Process an image (replace with actual image path)
        image_path = "map_two.png"  # Replace with your image path

        if Path(image_path).exists():
            results = pipeline.process_image(image_path)

            # Save point cloud
            pipeline.save_point_cloud(
                results["point_cloud"], "output_point_cloud.ply"
            )

            # Optional: Visualize point cloud
            pipeline.visualize_point_cloud(results["point_cloud"])

            print(
                f"Processing completed! Generated {results['metadata']['num_points']} 3D points"
            )
        else:
            logger.warning(f"Example image not found: {image_path}")

    except Exception as e:
        logger.error(f"Example execution failed: {e}")
