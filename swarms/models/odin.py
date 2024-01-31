import os
import supervision as sv
from ultralytics_example import YOLO
from tqdm import tqdm
from swarms.models.base_llm import AbstractLLM
from swarms.utils.download_weights_from_url import (
    download_weights_from_url,
)


class Odin(AbstractLLM):
    """
    Odin class represents an object detection and tracking model.

    Attributes:
        source_weights_path (str): The file path to the YOLO model weights.
        confidence_threshold (float): The confidence threshold for object detection.
        iou_threshold (float): The intersection over union (IOU) threshold for object detection.

    Example:
    >>> odin = Odin(
    ...     source_weights_path="yolo.weights",
    ...     confidence_threshold=0.3,
    ...     iou_threshold=0.7,
    ... )
    >>> odin.run(video="input.mp4")


    """

    def __init__(
        self,
        source_weights_path: str = "yolo.weights",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ):
        super(Odin, self).__init__()
        self.source_weights_path = source_weights_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        if not os.path.exists(self.source_weights_path):
            download_weights_from_url(
                url=source_weights_path,
                save_path=self.source_weights_path,
            )

    def run(self, video: str, *args, **kwargs):
        """
        Runs the object detection and tracking algorithm on the specified video.

        Args:
            video (str): The path to the input video file.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if the video was processed successfully, False otherwise.
        """
        model = YOLO(self.source_weights_path)

        tracker = sv.ByteTrack()
        box_annotator = sv.BoxAnnotator()
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video
        )
        video_info = sv.VideoInfo.from_video(video=video)

        with sv.VideoSink(
            target_path=self.target_video, video_info=video_info
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
