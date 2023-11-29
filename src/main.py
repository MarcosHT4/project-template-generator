from fastapi import (
    FastAPI, 
    UploadFile, 
    File, 
    HTTPException, 
    status,
    Depends

)
from starlette.middleware.cors import CORSMiddleware
from src.llm_service import TemplateLLM
from src.prompts import ProjectParams
from src.parsers import ProjectIdeas
from src.config import get_settings
import io
from PIL import Image
from src.detector import ObjectDetector, Detection
import numpy as np 
from functools import cache

_SETTINGS = get_settings()

app = FastAPI(
    title = _SETTINGS.service_name,
    version=_SETTINGS.k_revision,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_llm_service():
    return TemplateLLM()
@cache
def get_object_detector():
    print("creating model...")
    return ObjectDetector()

def predict_uploadfile(predictor, file, threshold):
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
            detail="Not an image"
        )
    # convertir a una imagen de Pillow
    img_obj = Image.open(img_stream)
    # crear array de numpy
    img_array = np.array(img_obj)
    return predictor.predict_image(img_array, threshold), img_array

@app.post("/objects")
def detect_objects(
    threshold: float = 0.5,
    file: UploadFile = File(...), 
    predictor: ObjectDetector = Depends(get_object_detector)
):
    results, _ = predict_uploadfile(predictor, file, threshold)
    
    return results


@app.post("/generate")
def generate_project(params: ProjectParams, service: TemplateLLM = Depends(get_llm_service)) -> ProjectIdeas:
    return service.generate(params)


@app.get("/")
def root():
    return {"status": "OK"}
