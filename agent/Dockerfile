# FROM nvidia/cuda:12.1.1-runtime-ubuntu22.0
FROM python:3.10-slim


WORKDIR /app

COPY requirements.txt /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip curl && \
    pip3 install gdown


RUN gdown "https://drive.google.com/uc?export=download&id=1WnUQsl2G-fG3DKyHAdAvqv6uCgoDWFqG" -O label_encoder.pkl
RUN gdown "https://drive.google.com/uc?export=download&id=1N0Nc9lRPxWzYqD-I5hXEyD5DHncH_ij_" -O movies_metadata.csv
RUN gdown "https://drive.google.com/uc?export=download&id=1idjzhR7is6G9bupS8YYkqh6tA066129q" -O vectorizer.pkl
RUN gdown "https://drive.google.com/uc?export=download&id=1yQNtQAIx90cAXPreoZ-w307tHouh_Wue" -O genre_model.pth

RUN pip3 install -r requirements.txt


COPY . /app
EXPOSE 5000 

CMD ["python3", "agent.py"]

