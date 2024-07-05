import os
import sys
import time
import argparse
import logging
from io import StringIO

from pyaml_env import parse_config

from operator import itemgetter
from operator import attrgetter
from itertools import groupby

import numpy as np
import pandas as pd

from flask import Flask, jsonify, request

import boto3

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

__author__ = "Miguel Salinas Gancedo"
__copyright__ = "Miguel Salinas Gancedo"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

# get environment variables from active profile            
if not os.getenv('ARG_PYTHON_PROFILES_ACTIVE'):
    _config = parse_config('./src/morphingprojections_backend_analytics/environment/environment.yaml')        
else:
    _config = parse_config('./src/morphingprojections_backend_analytics/environment/environment-' + os.getenv('ARG_PYTHON_PROFILES_ACTIVE') + '.yaml')

app = Flask(__name__)

_connection_minio = None

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

def connect_object_storage(config):
    return boto3.client('s3',
        endpoint_url="https://" + str(config["host"]) + ":" + str(config["port"]),
        aws_access_key_id=config["access_key"],
        aws_secret_access_key=config["secret_key"],
        verify=False,
        region_name='us-east-1')

def filter_datamatrix(bucket_datamatrix, file_datamatrix, view, sample_annotation, attribute_annotation, items):
    global _connection_minio

    #get headers
    headers_lst = _connection_minio.select_object_content(
        Bucket=bucket_datamatrix,
        Key=file_datamatrix,
        ExpressionType='SQL',
        Expression="SELECT * FROM s3object s LIMIT 1",
        InputSerialization = {'CSV': {"FileHeaderInfo": "NONE"}, 'CompressionType': 'NONE'},
        OutputSerialization = {'CSV': {}},
    )

    headers = None
    for event in headers_lst['Payload']:
        if 'Records' in event:
            records = event['Records']['Payload'].decode('utf-8')  

            df =  pd.read_csv(StringIO(records), header=None)
            
            if headers is None:
                headers = df
            else:            
                headers = pd.concat([headers, df])

        elif 'End' in event:
            print("End Event")
        elif 'Stats' in event:
            statsDetails = event['Stats']['Details']
            print("Stats details bytesScanned: ")
            print(statsDetails['BytesScanned'])
            print("Stats details bytesProcessed: ")
            print(statsDetails['BytesProcessed'])

    print(headers.head())    

    # get expressions
    if sample_annotation is not None:
        samples = ",".join(["'" + str(item) + "'" for item in items])
        samples = "(" + samples + ")"

        expression_lst = _connection_minio.select_object_content(
            Bucket=bucket_datamatrix,
            Key=file_datamatrix,
            ExpressionType='SQL',
            Expression="SELECT * FROM s3object s where s.\"sample_id\" IN " + samples,
            InputSerialization = {'CSV': {"FileHeaderInfo": "USE"}, 'CompressionType': 'NONE'},
            OutputSerialization = {'CSV': {}},
        )
    else:
        attributes = ",".join(["'" + str(item) + "'" for item in items])

        expression_lst = _connection_minio.select_object_content(
            Bucket=bucket_datamatrix,
            Key=file_datamatrix,
            ExpressionType='SQL',
            Expression="SELECT " + attributes + " FROM s3object s",
            InputSerialization = {'CSV': {"FileHeaderInfo": "USE"}, 'CompressionType': 'NONE'},
            OutputSerialization = {'CSV': {}},
        )              

    start = time.time()

    expressions = None
    for event in expression_lst['Payload']:
        if 'Records' in event:
            records = event['Records']['Payload'].decode('utf-8')  

            df =  pd.read_csv(StringIO(records), header=None)
            
            if expressions is None:
                expressions = df
            else:            
                expressions = pd.concat([expressions, df])

        elif 'End' in event:
            print("End Event")
        elif 'Stats' in event:
            statsDetails = event['Stats']['Details']
            print("Stats details bytesScanned: ")
            print(statsDetails['BytesScanned'])
            print("Stats details bytesProcessed: ")
            print(statsDetails['BytesProcessed'])

    end = time.time()
    print(end - start)

    expressions.columns = headers.iloc[0].values

    print(expressions.head())

    return expressions

@app.route('/')
def health():
    _logger.info("Health Endpoint")

    return jsonify('I am Ok')

@app.route('/analytics/tsne',  methods=['POST'])
def tsne():
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

    return response

@app.route('/analytics/histogram',  methods=['POST'])
def histogram():
    hist_bins = []

    # recover request data
    data = request.get_json() 

    # nesource metadata to be filter
    bucket_datamatrix = data['bucketDataMatrix']
    file_datamatrix = data['fileDataMatrix']

    # name and title chart
    name = data["name"]
    title = data["title"]             

    # view type selected: sample or attribute view and filter selected
    view = data["view"]
    sample_annotation = None
    if "filterSampleAnnotation" in data:   
        sample_annotation = data["filterSampleAnnotation"]

    attribute_annotation = None
    if "filterAttributeAnnotation" in data:           
        attribute_annotation = data["filterAttributeAnnotation"]  

    # number of bins selected
    bins = None        
    if "bins" in data:
        bins = data["bins"]

    # groups and items by grouped selected
    groups = data['groups']

    items = []
    for group in groups:
        items = items + group["values"]

    # get expression list from filters    
    _logger.info("Get filter items from data view %s for %s histogram", view, name)

    expressions = filter_datamatrix(bucket_datamatrix, file_datamatrix, view, sample_annotation, attribute_annotation, items)

    if view == "attribute_view":
        _logger.info("Get filter items from attribute_view: TODO")
            
    # group expressions
    for expression in expressions:
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
        hist_bins.append({
            "annotation": annotation[0],
            "group": annotation[1], 
            "color": annotation[2], 
            "value": int(expression_grouped_df.loc[annotation])
        })

   # parse respose to json list
    for annotation in expression_grouped_df.index:
        for group in groups:
            result = next((hist_bin for hist_bin in hist_bins if hist_bin["group"] == group["name"] and hist_bin["annotation"] == annotation[0]), None)
            
            if result is None:
                hist_bins.append({
                    "annotation": annotation[0],
                    "group": group["name"], 
                    "color": group["color"], 
                    "value": 0
                })            

    # order group respose by annotation
    hist_bins.sort(key=lambda item: item["annotation"])    

    return hist_bins

@app.route('/analytics/logistic_regression',  methods=['POST'])
def logistic_regression():
    response = []
    data = request.get_json() 

    # recover request data
    index_datamatrix = data['indexDataMatrix']
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

    expression_lst = filter_datamatrix(view, items, index_datamatrix, sample_annotation, attribute_annotations)

    # parse expression collection dataframe
    expressions = pd.json_normalize(expression_lst)

    expressions = expressions.pivot(index='sample_id', columns='attribute_id', values='value')
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
                "attribute_id": expressions.columns[index_d + 1], 
                "analytics_a": analytics_a,
                "analytics_b": analytics_b,
                "value": regression
            }) 

    response.sort(key=lambda x: x["value"], reverse=True)

    return response

def wsgi():
    # connect to elastic database
    _connection_minio = connect_object_storage(_config["minio"])

    return app

def main(args):
    global _connection_minio
    global _config

    args, unknown = parse_args(args)
    
    setup_logging(args.loglevel)
    
    # connect to elastic database
    _connection_minio = connect_object_storage(_config["minio"])

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
