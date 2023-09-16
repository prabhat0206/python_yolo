FROM python:3.10-slim-buster

WORKDIR /app

RUN curl --proto '=https' --tlsv1.2 -sSf -y https://sh.rustup.rs | sh
ENV PATH /home/pn/.cargo/bin:$PATH

COPY requirements.txt requirements.txt

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install timm==0.9.7
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

