"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = morphingprojections_backend_analytics.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import os
import argparse
import logging
import sys
import pandas as pd

from flask import Flask, jsonify, request
#from flask.ext.cors import CORS

from elasticsearch import Elasticsearch, ConnectionError, RequestError, NotFoundError, helpers

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

__author__ = "Miguel Salinas Gancedo"
__copyright__ = "Miguel Salinas Gancedo"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

# Project default configuration
PROJECT_ID = "genomic"
INDEX_DATAMATRIX = "dataset_datamatrix"
INDEX_SCROLL_MAX_TIME = '10s'
INDEX_MATCH_SIZE = 9000

# Database default configuration
#ELASTIC_HOST = "https://avib-elastic:9200"
ELASTIC_HOST = "https://avispe.edv.uniovi.es:443/kubernetes/elastic"
ELASTIC_USER = "elastic"
ELASTIC_PASSWORD = "password"
#ELASTIC_CERTIFICATE = os.path.dirname(os.path.realpath(__file__)) + "/certificates/ca-elastic-local.crt"
ELASTIC_CERTIFICATE = os.path.dirname(os.path.realpath(__file__)) + "/certificates/ca-elastic-avib.crt"

app = Flask(__name__)
#CORS(app)

# private module attributes
_index_name = PROJECT_ID + "_" + INDEX_DATAMATRIX
_connection_db = None

def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Analytics Backend services")
    parser.add_argument(
        "-p",
        "--port",
        dest="port",
        help="set server port",
        action="store",
        default=5000,
    )    
    parser.add_argument(
        "--log-level",
        dest="loglevel",
        help="set loglevel",
        action="store",
        default=logging.DEBUG,
    )

    return parser.parse_known_args()

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )

def connect_database(**kwargs):    
    # Create databse client connection and debug
    _connection_db = Elasticsearch(
        ELASTIC_HOST,
        basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD),
        ca_certs=ELASTIC_CERTIFICATE,
        verify_certs=True,
        http_compress=True
    )

    try:
        _logger.debug("Database info: %s", _connection_db.info())
    except ConnectionError as exception:
        _logger.error("Error database message %s from %s", exception.message, args.elastic_host)
        sys.exit()

    return _connection_db

@app.route('/')
def default_route():
    """Default route"""
    app.logger.debug('this is a DEBUG message')
    app.logger.info('this is an INFO message')
    app.logger.warning('this is a WARNING message')
    app.logger.error('this is an ERROR message')
    app.logger.critical('this is a CRITICAL message')

    _logger.info("INFO LOG")

    return jsonify('hello world')

@app.route('/tsne',  methods=['POST'])
def tsne():
    global _connection_db
    global _index_name

    response = []
    data = request.get_json() 

    # recover request data
    name = data['name']
    description = data['description']
    samples = data['samples']
    attributes = data['attributes']

    # filter by samples and attributes in database
    _logger.info("Create database filter to recover samples from elasticsearch for %s filter", name)

    filter = {
            "size": INDEX_MATCH_SIZE,
            "query": {
                "bool": {
                    "must": []
                }
            }
        }

    if len(samples) > 0:
        filter["query"]["bool"]["must"].append({"terms": {"sample_id": samples }})

    if len(attributes) > 0:
        filter["query"]["bool"]["must"].append({"terms": {"attribute": attributes }})

    _logger.info("Execute filter from elasticsearch in bulk requests")

    resp = _connection_db.search(
        index=_index_name, 
        body = filter,
        scroll = INDEX_SCROLL_MAX_TIME)

    # keep track of pass scroll _id
    old_scroll_id = resp['_scroll_id']

    # use a 'while' iterator to loop over document 'hits'
    expression_lst = []

    # iterate over the document hits for each 'scroll'
    while len(resp['hits']['hits']):        
        doc_count = 0
        for doc in resp['hits']['hits']:
            expression_lst.append(doc['_source'])

            doc_count += 1            

            _logger.info("DOC COUNT: %s", doc_count)

        _logger.info("TOTAL DOC COUNT: %s", doc_count)

        # make a request using the Scroll API
        resp = _connection_db.scroll(
            scroll_id = old_scroll_id,
            scroll = INDEX_SCROLL_MAX_TIME # length of time to keep search context
        )

        # check if there's a new scroll ID
        if old_scroll_id != resp['_scroll_id']:
            _logger.info("NEW SCROLL ID: %s", resp['_scroll_id'])

        # keep track of pass scroll _id
        old_scroll_id = resp['_scroll_id']

    # parse dataset to be projected
    _logger.info("Parse request to fit t-SNE model")

    df = pd.DataFrame(expression_lst)
    df = df.drop("sample_type", axis = 1)
    df = df.drop("cancer_code", axis = 1)
    
    dataset_expression = df.pivot(index='sample_id', columns='attribute', values='value')

    tsne = TSNE(perplexity=20, learning_rate=200, n_iter=500, n_components=2, method='barnes_hut', verbose=2, init='pca')

    dataset_projection = tsne.fit_transform(dataset_expression)
    dataset_projection_df = pd.DataFrame(dataset_projection, columns=['x', 'y'])

    # normalized between 0 and 1 the projection dataset
    _logger.info("Normalize t-SNE projection")

    scaler = MinMaxScaler()
    dataset_projection_df = pd.DataFrame(scaler.fit_transform(dataset_projection_df), columns=['x', 'y'])

    _logger.info("Parse Dataframe to json list")

    dataset_projection_lst = []
    for index in dataset_projection_df.index.values:                    
        dataset_projection_lst.append(
            {
                 "sample_id": dataset_expression.index[index],
                 "x": dataset_projection_df.iloc[index]["x"].item(),
                 "y": dataset_projection_df.iloc[index]["y"].item()
            }
        )

    _logger.info(len(dataset_projection_lst))

    return dataset_projection_lst

def main(args):
    global _connection_db

    args, unknown = parse_args(args)
    
    setup_logging(args.loglevel)
    
    # connect to elastic database
    _connection_db = connect_database()

    _logger.info("Starting service ...")
    app.run(host='0.0.0.0', port=args.port, debug=True)
    _logger.info("Service ends here")

    return app

def wsgi():
    global _connection_db

    # connect to elastic database
    _connection_db = connect_database()

    return app

def run():
    main(sys.argv[1:])
    
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    
    # set gunicorn logger to module logger
    _logger.handlers = gunicorn_logger.handlers
    _logger.setLevel(gunicorn_logger.level)

    # set gunicorn logger to flask logger
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == "__main__": 
    run()
