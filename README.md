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

# uniovi-avib-morphingprojections-backend-analytics

> Uniovi AVIB Morphing Projection Backend Analytic Service.

A longer description of your project goes here...

**STEP01**: Scaffolding your python project:

```
$ putup --markdown uniovi-avib-morphingprojections-backend-analytics \
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
```

**STEP04**: generate the requirements witl all python packages dependecies
```
$ pip list --format=freeze > requirements.txt
```

**STEP05**: Install in development mode your python package to test and debug locally

```
$ pip install -e .
```

**STEP06**: Manage docker images and run

```
$ docker build -t uniovi-avib-morphingprojections-backend-analytics .
$ docker run -p 5000:5000 uniovi-avib-morphingprojections-backend-analytics
```

**NOTE**: If you regenerate your requirements this local dependency will be added in the requirements file, 
be carefoul to not push this change

**STEP06**: Start service from gunicorn server
```
gunicorn --config gunicorn_config.py src.morphingprojections_backend_analytics.service:app
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
