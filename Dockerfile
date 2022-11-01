FROM python:3.8.10
RUN pip install --upgrade pip
WORKDIR /app
COPY . /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip --no-cache-dir install -r requirements.txt
CMD ["python3", "app.py"]
