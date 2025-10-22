FROM python:3.12-slim

# Python boot profile
ARG ARG_PYTHON_PROFILES_ACTIVE
ENV PYTHON_PROFILES_ACTIVE=${ARG_PYTHON_PROFILES_ACTIVE}

RUN apt-get update && apt-get install -y git

COPY requirements.txt /

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY ./src /app/src
COPY ./gunicorn_config.py /app

ENV PYTHONUNBUFFERED=1

WORKDIR /app

EXPOSE 5000

CMD ["gunicorn","--config", "gunicorn_config.py", "--log-level=debug", "src.morphingprojections_backend_analytics.service:wsgi()"]