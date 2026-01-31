FROM python:3.9

WORKDIR /app




COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
