FROM python:3.11
ARG OPENAI_KEY
ENV OPENAI_KEY=$OPENAI_KEY
ENV PORT 8000

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt /
RUN pip install -r requirements.txt
RUN curl -O -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt


COPY ./src /src
# COPY ./yolov8n.pt ./yolov8n.pt
#COPY .env /.env


CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT}
