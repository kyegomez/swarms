import numpy as np
import supervision as sv


class MarkVisualizer:
    """
    A class for visualizing different marks including bounding boxes, masks, polygons,
    and labels.

    Parameters:
        line_thickness (int): The thickness of the lines for boxes and polygons.
        mask_opacity (float): The opacity level for masks.
        text_scale (float): The scale of the text for labels.
    """

    def __init__(
        self,
        line_thickness: int = 2,
        mask_opacity: float = 0.1,
        text_scale: float = 0.6,
    ) -> None:
        self.box_annotator = sv.BoundingBoxAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=line_thickness,
        )
        self.mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX, opacity=mask_opacity
        )
        self.polygon_annotator = sv.PolygonAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=line_thickness,
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.Color.black(),
            text_color=sv.Color.white(),
            color_lookup=sv.ColorLookup.INDEX,
            text_position=sv.Position.CENTER_OF_MASS,
            text_scale=text_scale,
        )

    def visualize(
        self,
        image: np.ndarray,
        marks: sv.Detections,
        with_box: bool = False,
        with_mask: bool = False,
        with_polygon: bool = True,
        with_label: bool = True,
    ) -> np.ndarray:
        """
        Visualizes annotations on an image.

        This method takes an image and an instance of sv.Detections, and overlays
        the specified types of marks (boxes, masks, polygons, labels) on the image.

        Parameters:
            image (np.ndarray): The image on which to overlay annotations.
            marks (sv.Detections): The detection results containing the annotations.
            with_box (bool): Whether to draw bounding boxes. Defaults to False.
            with_mask (bool): Whether to overlay masks. Defaults to False.
            with_polygon (bool): Whether to draw polygons. Defaults to True.
            with_label (bool): Whether to add labels. Defaults to True.

        Returns:
            np.ndarray: The annotated image.
        """
        annotated_image = image.copy()
        if with_box:
            annotated_image = self.box_annotator.annotate(
                scene=annotated_image, detections=marks
            )
        if with_mask:
            annotated_image = self.mask_annotator.annotate(
                scene=annotated_image, detections=marks
            )
        if with_polygon:
            annotated_image = self.polygon_annotator.annotate(
                scene=annotated_image, detections=marks
            )
        if with_label:
            labels = list(map(str, range(len(marks))))
            annotated_image = self.label_annotator.annotate(
                scene=annotated_image, detections=marks, labels=labels
            )
        return annotated_image
