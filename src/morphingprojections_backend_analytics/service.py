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

from operator import itemgetter
from operator import attrgetter
from itertools import groupby

import numpy as np
import pandas as pd

from flask import Flask, jsonify, request

from elasticsearch import Elasticsearch, ConnectionError, RequestError, NotFoundError, helpers

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

__author__ = "Miguel Salinas Gancedo"
__copyright__ = "Miguel Salinas Gancedo"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

# Project default configuration
PROJECT_ID = "genomic"
#INDEX_DATAMATRIX = "dataset_datamatrix"
INDEX_DATAMATRIX = "dataset_datamatrix_new"
INDEX_DATASET_SAMPLE_VIEW = "dataset_encoding_default"
INDEX_DATASET_ATTRIBUTE_VIEW = "dataset_attribute_view_encoding_default"
INDEX_SCROLL_MAX_TIME = '10s'
INDEX_MATCH_SIZE = 10000

# Database default configuration
ELASTIC_HOST = "https://avib-elastic:9200"
#ELASTIC_HOST = "https://avispe.edv.uniovi.es:443/kubernetes/elastic"
ELASTIC_USER = "elastic"
ELASTIC_PASSWORD = "password"
ELASTIC_CERTIFICATE = os.path.dirname(os.path.realpath(__file__)) + "/certificates/ca-elastic-local.crt"
#ELASTIC_CERTIFICATE = os.path.dirname(os.path.realpath(__file__)) + "/certificates/ca-elastic-avib.crt"

app = Flask(__name__)

# private module attributes
_index_datamatrix = PROJECT_ID + "_" + INDEX_DATAMATRIX
_index_dataset_sample_view = PROJECT_ID + "_" + INDEX_DATASET_SAMPLE_VIEW
_index_dataset_attribute_view = PROJECT_ID + "_" + INDEX_DATASET_ATTRIBUTE_VIEW

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
        #ca_certs=ELASTIC_CERTIFICATE,
        #verify_certs=True,
        verify_certs=False,
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

def filter_index_datamatrix(view, items, filter_samples=None, filter_attributes=None):
    global _connection_db
    global _index_datamatrix

    # convert the sample and attribute annotation to collection to be general request
    if filter_samples is not None and type(filter_samples) != list:
        filter_samples = [filter_samples]

    if filter_attributes is not None and type(filter_attributes) != list:
        filter_attributes = [filter_attributes]

    doc_total = 0
    response = []    

    body = {
            "size": INDEX_MATCH_SIZE,
            "query": {
                "bool": {
                    "must": []
                }
            },
            "fields": [
                "sample_id",
                "attribute",
                "value",
            ]
        }

    if view == "sample_view":
        body["query"]["bool"]["must"].append({"terms": {"sample_id": items }})
        #body["query"]["bool"]["must"].append({"term": {"data_type": "mirna_expression" }})

        if (filter_attributes is not None):
            body["query"]["bool"]["must"].append({"terms": {"attribute": filter_attributes }})

    if view == "attribute_view":
        body["query"]["bool"]["must"].append({"terms": {"attribute": items }})

        if (filter_samples is not None):
            body["query"]["bool"]["must"].append({"terms": {"sample_id": filter_samples }})

    resp = _connection_db.search(
        index=_index_datamatrix, 
        body = body,
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

            #_logger.info("DOC COUNT: %s", doc_count)

        #_logger.info("TOTAL DOC COUNT: %s", doc_count)

        doc_total = doc_total + doc_count

        #_logger.info("TOTAL DOC: %s", doc_total)

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

    return expression_lst

def filter_index_sample_view(items):
    global _connection_db
    global _index_dataset_sample_view

    response = []

    body = {
            "size": INDEX_MATCH_SIZE,
            "query": {
                "terms": {
                    "sample_id": items
                }
            }
        }

    resp = _connection_db.search(
        index=_index_dataset_sample_view, 
        body = body,
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

    return expression_lst

def filter_index_attribute_view(items):
    global _connection_db
    global _index_dataset_attribute_view

    response = []

    body = {
            "size": INDEX_MATCH_SIZE,
            "query": {
                "terms": {
                    "attribute": items
                }
            }
        }

    resp = _connection_db.search(
        index=_index_dataset_attribute_view, 
        body = body,
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

    return expression_lst

@app.route('/analytics/tsne',  methods=['POST'])
def tsne():
    global _connection_db
    global _index_datamatrix

    response = []
    data = request.get_json() 

    # recover request data
    name = data['name']
    title = data['title']
    attributes = data['attributes']

    samples = None
    if "samples" in data:
        samples = data["samples"]

    # create database filter    
    _logger.info("Create database body to recover samples from elasticsearch for %s filter", name)

    body = {
            "size": INDEX_MATCH_SIZE,
            "query": {
                "bool": {
                    "must": []
                }
            }
        }

    if len(samples) > 0:
        body["query"]["bool"]["must"].append({"terms": {"sample_id": samples }})

    if len(attributes) > 0:
        body["query"]["bool"]["must"].append({"terms": {"attribute": attributes }})

    # execute database filter  
    _logger.info("Execute filter from elasticsearch in bulk requests")

    resp = _connection_db.search(
        index=_index_datamatrix, 
        body = body,
        scroll = INDEX_SCROLL_MAX_TIME)

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
    #df = df.drop("sample_type", axis = 1)
    df = df.drop("data_type", axis = 1)
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

@app.route('/analytics/histogram',  methods=['POST'])
def histogram():
    histBins = []
    data = request.get_json() 

    # recover request data
    name = data["name"]
    title = data["title"] 
    view = data["view"]
    groups = data['groups']

    sample_annotation = None
    if "filterSampleAnnotation" in data:   
        sample_annotation = data["filterSampleAnnotation"]

    attribute_annotation = None
    if "filterAttributeAnnotation" in data:           
        attribute_annotation = data["filterAttributeAnnotation"]  

    bins = None        
    if "bins" in data:
        bins = data["bins"]

    # aggregate all items grouped
    items = []
    for group in groups:
        items = items + group["values"]

    # get expression list from filters    
    _logger.info("Get filter items from data view %s for %s histogram", view, name)

    expression_lst = None
    if view == "sample_view":
        if attribute_annotation is None:
            expression_lst = filter_index_sample_view(items)         
        else: 
            expression_lst = filter_index_datamatrix(view, items, sample_annotation, attribute_annotation)

    if view == "attribute_view":
        if sample_annotation is None:
            expression_lst = filter_index_attribute_view(items)
        else: 
            expression_lst = filter_index_datamatrix(view, items, sample_annotation, attribute_annotation)
            
    # group expressions
    for expression in expression_lst:
        for group in groups:
            for value in group["values"]:
                if value == expression["sample_id"]:
                    expression["group"] = group["name"]
                    expression["color"] = group["color"]

                    break

    # calculate histogram from expressions
    _logger.info("Group filter items from data view %s for %s histogram", view, name)

    if sample_annotation is not None:
        expression_df = pd.DataFrame(expression_lst)

        expression_grouped_df = expression_df.groupby([sample_annotation, "group", "color"])[sample_annotation].count()
    else:
        max_expression = max(expression_lst, key=lambda exp: exp["value"])
        min_expression = min(expression_lst, key=lambda exp: exp["value"])        
        interval = (max_expression["value"] - min_expression["value"]) / bins

        for expression in expression_lst:
            for bin in range(len(expression_lst)):
                if (expression["value"] < min_expression["value"] + interval * bin):
                    expression["bin"] = min_expression["value"] + bin * interval

                    break

        expression_df = pd.DataFrame(expression_lst)
        expression_df = expression_df.round(3)

        expression_grouped_df = expression_df.groupby(["bin", "group", "color"])["bin"].count()

    # parse respose to json list
    for annotation in expression_grouped_df.index:            
        histBins.append({
            "annotation": annotation[0],
            "group": annotation[1], 
            "color": annotation[2], 
            "value": int(expression_grouped_df.loc[annotation])
        })

   # parse respose to json list
    for annotation in expression_grouped_df.index:
        for group in groups:
            result = next((histBin for histBin in histBins if histBin["group"] == group["name"] and histBin["annotation"] == annotation[0]), None)
            
            if result is None:
                histBins.append({
                    "annotation": annotation[0],
                    "group": group["name"], 
                    "color": group["color"], 
                    "value": 0
                })            

    # order group respose by annotation
    histBins.sort(key=lambda item: item["annotation"])    

    return histBins

@app.route('/analytics/logistic_regression',  methods=['POST'])
def logistic_regression():
    response = []
    data = request.get_json() 

    # recover request data
    name = data["name"]
    title = data["title"] 
    view = data["view"]
    groups = data['groups']

    sample_annotation = None
    if "filterSampleAnnotation" in data:   
        sample_annotation = data["filterSampleAnnotation"]

    attribute_annotation = None
    if "filterAttributeAnnotation" in data:           
        attribute_annotation = data["filterAttributeAnnotation"] 

    # aggregate all items grouped    
    items = []
    for index, group in  enumerate(groups):
        items = items + group["values"]

    sample_group_lst = []
    for index_group, group in enumerate(groups):
        for sample_id in group["values"]:
            sample_group_lst.append({'sample_id': sample_id, 'group_id': index_group})        

    # get expression list from filters only by miRNA attributes
    attributes = data['attributes']

    attribute_annotations = []
    for attribute in attributes:
        if "MIMAT" in attribute["key"]:
            attribute_annotations.append(attribute["key"])

    # get expressions from db
    _logger.info("Get expressions from datamatrix for view %s and %s to execute logistic regression", view, name)

    expression_lst = filter_index_datamatrix(view, items, sample_annotation, attribute_annotations)

    # parse expression collection dataframe
    expressions = pd.json_normalize(expression_lst)

    expressions = expressions.pivot(index='sample_id', columns='attribute', values='value')
    expressions = expressions.reset_index()

    # parse group collection to dataframe
    sample_groups = pd.json_normalize(sample_group_lst)

    # merge expression with group dataframes
    expressions = expressions.merge(sample_groups, how='inner', on='sample_id')
    expressions = expressions.sort_values('group_id', ascending=True,)

    # get logistic regression subdataframes to be trained
    X = expressions.iloc[:,1:-1]
    y = expressions.loc[:,"group_id"]

    LR = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(X, y)

    # get logistic regression coeficients (normal vector hyperplane)
    d = LR.coef_
    
    # normalize logistic regression results
    d = np.abs(d) # set absolute values
    d = d/np.max(np.max(d)) # Normalize dataframe
    d = d[0]
    
    response = []
    for index_d, regression in enumerate(d):
        sub_expression = expressions[[expressions.columns[index_d + 1], "group_id"]]
        mean_expressions = sub_expression.groupby(by=["group_id"]).mean()
        standard_deviation_expressions = sub_expression.groupby(by=["group_id"]).std()
        
        analytics_a = str(round(mean_expressions.values[0][0], 2)) + "±" + str(round(standard_deviation_expressions.values[0][0], 2))
        analytics_b = str(round(mean_expressions.values[1][0], 2)) + "±" + str(round(standard_deviation_expressions.values[1][0], 2))

        response.append(
            {
                "attribute": expressions.columns[index_d + 1], 
                "analytics_a": analytics_a,
                "analytics_b": analytics_b,
                "value": regression
            }) 

    response.sort(key=lambda x: x["value"], reverse=True)

    return response

def wsgi():
    global _connection_db

    # connect to elastic database
    _connection_db = connect_database()

    return app

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
