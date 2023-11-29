from ultralytics import YOLO
from pydantic import BaseModel
from enum import Enum
from src.config import get_settings

SETTINGS = get_settings()

class PredictionType(str, Enum):
    classification = "CLS"
    object_detection = "OD"
    segmentation = "SEG"

class GeneralPrediction(BaseModel):
    pred_type: PredictionType

class Detection(GeneralPrediction):
    number_of_persons: int

class ObjectDetector:
    def __init__(self) -> None:
        self.model = YOLO(SETTINGS.yolo_version)

    def predict_image(self, image_array, threshold):
        results = self.model(image_array, conf=threshold)[0]
        labels = [results.names[i] for i in results.boxes.cls.tolist()]

        boxes = [[int(v) for v in box] for box in results.boxes.xyxy.tolist()]
        
        # Return only the person count

        count = labels.count("person")

        detection = Detection(pred_type=PredictionType.object_detection, number_of_persons=count)

        return detection