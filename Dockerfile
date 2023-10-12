FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS builder

WORKDIR /workspace

RUN apt-get update && apt-get install -y python3 python3-pip python3-dev libglib2.0-0 libsm6 libxrender1 libxext6 uvicorn ffmpeg libsm6 libxext6 

RUN pip3 install --upgrade pip
RUN pip3 install fastapi ultralytics

COPY . .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
