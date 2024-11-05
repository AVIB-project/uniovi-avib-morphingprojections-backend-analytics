<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/uniovi-avib-morphingprojections-backend-analytics.svg?branch=main)](https://cirrus-ci.com/github/<USER>/uniovi-avib-morphingprojections-backend-analytics)
[![ReadTheDocs](https://readthedocs.org/projects/uniovi-avib-morphingprojections-backend-analytics/badge/?version=latest)](https://uniovi-avib-morphingprojections-backend-analytics.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/uniovi-avib-morphingprojections-backend-analytics/main.svg)](https://coveralls.io/r/<USER>/uniovi-avib-morphingprojections-backend-analytics)
[![PyPI-Server](https://img.shields.io/pypi/v/uniovi-avib-morphingprojections-backend-analytics.svg)](https://pypi.org/project/uniovi-avib-morphingprojections-backend-analytics/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/uniovi-avib-morphingprojections-backend-analytics.svg)](https://anaconda.org/conda-forge/uniovi-avib-morphingprojections-backend-analytics)
[![Monthly Downloads](https://pepy.tech/badge/uniovi-avib-morphingprojections-backend-analytics/month)](https://pepy.tech/project/uniovi-avib-morphingprojections-backend-analytics)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/uniovi-avib-morphingprojections-backend-analytics)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# Description

> Uniovi AVIB Morphing Projection Backend Analytic Service.

**STEP01**: Scaffolding your python project:

```
$ putup --markdown uniovi-avib-morphingprojections-backend-analytics -p morphingprojections_backend_analytics \
      -d "Uniovi AVIB Morphing Projection Backend Analytic Service." \
      -u https://gsdpi@dev.azure.com/gsdpi/avib/_git/uniovi-avib-morphingprojections-backend-analytics
```

**STEP02**: Create a virtual environment in your python project and activated it:

```
$ cd uniovi-avib-morphingprojections-backend-analytics

$ python3 -m venv .venv 

$ source .venv/bin/activate
(.venv) miguel@miguel-Inspiron-5502:~/git/uniovi/uniovi-avib-morphingprojections-backend-analytics$
```

**STEP03**: Install development and business dependencies in your project

```
$ pip install tox
$ pip install numpy
$ pip install pandas
$ pip install scikit-learn
$ pip install seaborn
$ pip install flask
$ pip install elasticsearch
```

**STEP04**: generate the requirements with all python package dependencies
```
$ pip freeze > requirements.txt
```

**STEP06**: Manage python project
```
tox -e docs  # to build your documentation
tox -e build  # to build your package distribution
```

**STEP07**: Start service from gunicorn server locally
```
gunicorn --config gunicorn_config.py --log-level=debug 'src.morphingprojections_backend_analytics.service:wsgi()'
```

**STEP08**: Manage docker images and run

build image for local environment:

```
docker build -t uniovi-avib-morphingprojections-backend-analytics:1.0.0 .

docker tag uniovi-avib-morphingprojections-backend-analytics:1.0.0 avibdocker.azurecr.io/uniovi-avib-morphingprojections-backend-analytics:1.0.0

docker push uniovi-avib-morphingprojections-backend-analytics:1.0.0
```

build image for local minikube environment:

```
docker build --build-arg ARG_PYTHON_PROFILES_ACTIVE=minikube -t uniovi-avib-morphingprojections-backend-analytics:1.0.0 .

docker tag uniovi-avib-morphingprojections-backend-analytics:1.0.0 avibdocker.azurecr.io/uniovi-avib-morphingprojections-backend-analytics:1.0.0

docker push avibdocker.azurecr.io/uniovi-avib-morphingprojections-backend-analytics:1.0.0
```

build image for avib environment:

```
docker build --build-arg ARG_PYTHON_PROFILES_ACTIVE=avib -t uniovi-avib-morphingprojections-backend-analytics:1.0.0 .

docker tag uniovi-avib-morphingprojections-backend-analytics:1.0.0 avibdocker.azurecr.io/uniovi-avib-morphingprojections-backend-analytics:1.0.0

docker push avibdocker.azurecr.io/uniovi-avib-morphingprojections-backend-analytics:1.0.0
```

Execute flow locally for a case_id 65cdc989fa8c8fdbcefac01e:

```
docker run --rm uniovi-avib-morphingprojections-backend-analytics:1.0.0
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
