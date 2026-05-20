"""Ultralytics YOLO detector adapter."""

from pathlib import Path
from typing import Any

from multimodalcv.core.models import BoundingBox, Detection, ObjectClass
from multimodalcv.detection.base import ObjectDetector
from multimodalcv.video.reader import VideoFrame


class YOLODependencyError(ImportError):
    """Raised when the optional YOLO dependency is not installed."""


class YOLODetector(ObjectDetector):
    """Object detector backed by an Ultralytics YOLO model."""

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        *,
        confidence_threshold: float = 0.25,
        supported_classes: set[ObjectClass] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise YOLODependencyError(
                "YOLO support requires the optional dependency: pip install -e .[yolo]"
            ) from error

        self._model = YOLO(str(model_path))
        self._confidence_threshold = confidence_threshold
        self._supported_classes = supported_classes or {ObjectClass.PERSON, ObjectClass.CAR}

    def detect(self, frame: VideoFrame) -> list[Detection]:
        results = self._model.predict(frame.image, verbose=False)
        if not results:
            return []

        detections: list[Detection] = []
        result = results[0]
        names = result.names

        for box in result.boxes:
            detection = _box_to_detection(
                box=box,
                names=names,
                frame=frame,
                confidence_threshold=self._confidence_threshold,
                supported_classes=self._supported_classes,
            )
            if detection is not None:
                detections.append(detection)

        return detections


def _box_to_detection(
    *,
    box: Any,
    names: dict[int, str],
    frame: VideoFrame,
    confidence_threshold: float,
    supported_classes: set[ObjectClass],
) -> Detection | None:
    confidence = float(box.conf[0])
    if confidence < confidence_threshold:
        return None

    class_index = int(box.cls[0])
    object_class = _map_yolo_class(names[class_index])
    if object_class is None or object_class not in supported_classes:
        return None

    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
    return Detection(
        frame_index=frame.frame_index,
        timestamp_sec=frame.timestamp_sec,
        object_class=object_class,
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def _map_yolo_class(class_name: str) -> ObjectClass | None:
    normalized = class_name.casefold()
    if normalized == "person":
        return ObjectClass.PERSON
    if normalized == "car":
        return ObjectClass.CAR
    return None

