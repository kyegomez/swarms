import supervision as sv
from ultraanalytics import YOLO
from tqdm import tqdm
from swarms.models.base_llm import AbstractLLM

class Odin(AbstractLLM):
    """
    Odin class represents an object detection and tracking model.

    Args:
        source_weights_path (str): Path to the weights file for the object detection model.
        source_video_path (str): Path to the source video file.
        target_video_path (str): Path to save the output video file.
        confidence_threshold (float): Confidence threshold for object detection.
        iou_threshold (float): Intersection over Union (IoU) threshold for object detection.

    Attributes:
        source_weights_path (str): Path to the weights file for the object detection model.
        source_video_path (str): Path to the source video file.
        target_video_path (str): Path to save the output video file.
        confidence_threshold (float): Confidence threshold for object detection.
        iou_threshold (float): Intersection over Union (IoU) threshold for object detection.
    """

    def __init__(
        self,
        source_weights_path: str = None,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ):
        super(Odin, self).__init__()
        self.source_weights_path = source_weights_path
        self.target_video_path = target_video_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def run(self, video_path: str, *args, **kwargs):
        """
        Runs the object detection and tracking algorithm on the specified video.

        Args:
            video_path (str): The path to the input video file.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if the video was processed successfully, False otherwise.
        """
        model = YOLO(self.source_weights_path)

        tracker = sv.ByteTrack()
        box_annotator = sv.BoxAnnotator()
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )
        video_info = sv.VideoInfo.from_video_path(
            video_path=video_path
        )

        with sv.VideoSink(
            target_path=self.target_video_path, video_info=video_info
        ) as sink:
            for frame in tqdm(
                frame_generator, total=video_info.total_frames
            ):
                results = model(
                    frame,
                    verbose=True,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                )[0]
                detections = sv.Detections.from_ultranalytics(results)
                detections = tracker.update_with_detections(
                    detections
                )

                labels = [
                    f"#{tracker_id} {model.model.names[class_id]}"
                    for _, _, _, class_id, tracker_id in detections
                ]

                annotated_frame = box_annotator.annotate(
                    scene=frame.copy(),
                    detections=detections,
                    labels=labels,
                )

                result = sink.write_frame(frame=annotated_frame)
                return result
