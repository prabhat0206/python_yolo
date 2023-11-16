FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS builder

WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Uncomment and use these lines to install additional dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6
# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# ENV PATH /home/pn/.cargo/bin:$PATH

COPY . .

# Install Python dependencies
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# COPY similarity /app/
# COPY server.py /app/


EXPOSE 8090

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]
