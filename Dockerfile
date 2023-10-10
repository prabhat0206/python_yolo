FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS builder

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev

COPY requirements.txt /workspace/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision
RUN pip3 install timm
RUN pip3 install -r requirements.txt

RUN apt install uvicorn -y
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
