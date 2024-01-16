FROM python:3.8-slim

RUN apt-get update && apt-get install -y git

COPY requirements.txt /

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY ./src /app/src
COPY ./gunicorn_config.py /app

WORKDIR /app

EXPOSE 5000

CMD ["gunicorn","--config", "gunicorn_config.py", "--log-level=debug", "src.morphingprojections_backend_analytics.service:wsgi()"]