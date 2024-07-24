import os
import sys
import time
import ast
import argparse
import logging
from io import StringIO
from io import BytesIO
import json
import shelve

from pyaml_env import parse_config

from operator import itemgetter
from operator import attrgetter
from itertools import groupby

import numpy as np
import pandas as pd

from flask import Flask, jsonify, request

import boto3

from minio import Minio
from minio.select import (ParquetInputSerialization, CSVInputSerialization, CSVOutputSerialization, SelectRequest)

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

__author__ = "Miguel Salinas Gancedo"
__copyright__ = "Miguel Salinas Gancedo"
__license__ = "MIT"

CACHE_FOLDER = "./src/morphingprojections_backend_analytics/.cache"

_logger = logging.getLogger(__name__)

# get environment variables from active profile            
if not os.getenv('ARG_PYTHON_PROFILES_ACTIVE'):
    _config = parse_config('./src/morphingprojections_backend_analytics/environment/environment.yaml')        
else:
    _config = parse_config('./src/morphingprojections_backend_analytics/environment/environment-' + os.getenv('ARG_PYTHON_PROFILES_ACTIVE') + '.yaml')

app = Flask(__name__)

_client_minio = None

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
        endpoint_url=str(config["scheme"]) + "://" + str(config["host"]) + ":" + str(config["port"]),
        aws_access_key_id=config["access_key"],
        aws_secret_access_key=config["secret_key"],
        verify=False,
        region_name='us-east-1')    

    '''
    return Minio(
        str(config["host"]) + ":" + str(config["port"]),
        access_key=config["access_key"],
        secret_key=config["secret_key"],
        secure=False
    )
    '''       

def filter_object_storage(bucket, key, expression, fileHeaderInfo="USE", compressionType="NONE", fileFormat="CSV"):
    start = time.time()

    if fileFormat == "CSV":
        result = _client_minio.select_object_content(
            Bucket=bucket,
            Key=key,
            ExpressionType='SQL',
            Expression=expression,            
            InputSerialization = {
                'CSV': {"FileHeaderInfo": fileHeaderInfo}, 'CompressionType': compressionType},
            OutputSerialization = {
                'CSV': {}
            },
        )   

        '''
        result = _client_minio.select_object_content(
            bucket,
            key,
            SelectRequest(
                expression,
                CSVInputSerialization(file_header_info=fileHeaderInfo),
                CSVOutputSerialization()
            )   
        )    
        '''           
    elif fileFormat == "Parquet":
        result = _client_minio.select_object_content(
            Bucket=bucket,
            Key=key,
            ExpressionType='SQL',
            Expression=expression,            
            InputSerialization = {
                'Parquet': {}
            },
            OutputSerialization = {
                'CSV': {}
            },
        )

        '''
        result = _client_minio.select_object_content(
            bucket,
            key,
            SelectRequest(
                expression,
                ParquetInputSerialization(),
                CSVOutputSerialization()
            )   
        )  
        ''' 

    '''
    records = ""
    for data in result.stream():
       records += data.decode()
    '''
    
    # concatenate final string with all rows selected
    records = ""
    #records = []
    for event in result['Payload']:
        if 'Records' in event:
            events = event['Records']['Payload'].decode('UTF-8')            
            #events =  event['Records']['Payload'].decode('ISO-8859-1')
            records += events            
        elif 'Stats' in event:
            stats = event['Stats']['Details']
        elif 'Cont' in event:
            cont = event['Cont']

    if fileHeaderInfo == "USE":
        # parse string concatenated to dataframe for rows selected
        records_df = pd.read_csv(StringIO(records), header=None)
    else:
        # parse string concatenated to dataframe for header selected
        records_df = pd.read_csv(StringIO(records))  

    # timestamp track
    end = time.time()
    _logger.info(end - start)

    return records_df

def filter_datamatrix(bucket, key, view, items):
    # get datamatrix dataframe header
    #headers_df = filter_object_storage(bucket=bucket, key=key, expression="SELECT * FROM s3object LIMIT 1", fileHeaderInfo="NONE")    
    headers_df = filter_object_storage(bucket=bucket, key=key, expression="SELECT * FROM s3object LIMIT 1", fileHeaderInfo="NONE", fileFormat="Parquet")    

    # get datamatrix expressions dataframe filtered by items (sample view (primal) or attribute view (dual))
    if view == "sample_view":
        samples = "(" + ",".join(["'" + str(item) + "'" for item in items]) + ")"

        expression_df = filter_object_storage(bucket=bucket, key=key, expression="SELECT * FROM s3object WHERE sample_id IN " + samples)
        #expression_df = filter_object_storage(bucket=bucket, key=key, expression="SELECT * FROM s3object WHERE sample_id IN " + samples, fileFormat="Parquet")

        # add datamatrix header to expressions dataframe
        expression_df.columns = headers_df.columns.values        
    else:
        attributes = ",".join(["'" + str(item) + "'" for item in items])

        expression_df = filter_object_storage(bucket=bucket, key=key, expression="SELECT " + attributes + " FROM s3object")
        #expression_df = filter_object_storage(bucket=bucket, key=key, expression="SELECT " + attributes + " FROM s3object", fileFormat="Parquet")

        # add datamatrix headers to expressions dataframe
        expression_df.columns = headers_df.columns.values

        # set transposed dataframe
        expression_df = expression_df.T

    return expression_df

def filter_cache_datamatrix(bucket, key, view, items):
    # get file name from key. The bucker is the organizationId and the key has this structure: projectId/caseId/fileName
    keys = key.split('/')
    file_name = keys[2]

    # get file last modify metadata to create a unique file key to be cached
    response = _client_minio.head_object(Bucket=bucket, Key=key)
    datetime_value = response["LastModified"]
    file_key = file_name + "_" + datetime_value.strftime("%Y%m%d%H%M%S")

     # create a cache db per organization
    cache_db = "cache_" + bucket + ".db"

    # Check if the file key is already in the cache
    with shelve.open(CACHE_FOLDER + "/" + cache_db) as cache:        
        if file_key not in cache:
            # download file from minio and cache locally
            file_stream = _client_minio.get_object(Bucket=bucket, Key=key)
            file_data = file_stream['Body'].read() 

            cache[file_key] = file_data

            _logger.info("Reading from file " + file_key)
        else:
            # If in the cache, retrieve the content from the cache
            _logger.info("Reading from cache " + file_key)
 
        # parse parquet cache file to dataframe
        df_datamatrix = pd.read_parquet(BytesIO(cache[file_key]))

        # Filter datamatrix dataframe filtered by items (sample view (primal) or attribute view (dual))
        if view == "sample_view":
            # filter datamatrix dataframe rows by sample_id
            df_datamatrix = df_datamatrix[df_datamatrix["sample_id"].isin(items)]
        else:
            # filter datamatrix dataframe columns by attribute_id
            df_datamatrix = df_datamatrix[items]

            # transposed datamatrix dataframe
            df_datamatrix = df_datamatrix.T

        # Return the content either from the cache or newly read file
        return df_datamatrix

def filter_annotation(bucket, file_annotation):
    # get annotation dataframe headers
    headers_df = filter_object_storage(bucket=bucket, key=file_annotation, expression="SELECT * FROM s3object LIMIT 1", fileHeaderInfo="NONE")

    # get annotations dataframe
    annotation_df = filter_object_storage(bucket, file_annotation, "SELECT * FROM s3object s")

    # add resource headers to filtered expressions
    annotation_df.columns = headers_df.columns.values

    return annotation_df

def filter_cache_annotation(bucket, key):
# get file name from key. The bucker is the organizationId and the key has this structure: projectId/caseId/fileName
    keys = key.split('/')
    file_name = keys[2]

    # get file last modify metadata to create a unique file key to be cached
    response = _client_minio.head_object(Bucket=bucket, Key=key)
    datetime_value = response["LastModified"]
    file_key = file_name + "_" + datetime_value.strftime("%Y%m%d%H%M%S")

     # create a cache db per organization
    cache_db = "cache_" + bucket + ".db"

    # Check if the file key is already in the cache
    with shelve.open(CACHE_FOLDER + "/" + cache_db) as cache:        
        if file_key not in cache:
            # download file from minio and cache locally
            file_stream = _client_minio.get_object(Bucket=bucket, Key=key)
            file_data = file_stream['Body'].read() 

            cache[file_key] = file_data

            _logger.info("Reading from file " + file_key)
        else:
            # If in the cache, retrieve the content from the cache
            _logger.info("Reading from cache " + file_key)
 
        # parse parquet cache file to dataframe
        df_annotation = pd.read_parquet(BytesIO(cache[file_key]))

        return df_annotation

@app.route('/')
def health():
    _logger.info("Health Endpoint")

    return jsonify('I am Ok')

@app.route('/analytics/tsne', methods=['POST'])
def tsne():
    start = time.time()

    response = []
    data = request.get_json() 

    # recover request data
    name = data['name']
    title = data['title']

    samples = None
    if "samples" in data:
        samples = data["samples"]

    attributes = None
    if "attributes" in data:
        attributes = data['attributes']

    # TODO

    # timestamp track
    end = time.time()
    _logger.info("TSNE processing time: " + str(end - start))

    return response

@app.route('/analytics/histogram', methods=['POST'])
def histogram():
    start = time.time()

    hist_bins = []

    # recover request json data
    data = request.get_json() 

    # get resource files
    bucket_datamatrix = data['bucketDataMatrix']
    file_datamatrix = data['fileDataMatrix']
    if "bucketSampleAnnotation" in data:  
        bucket_sample_annotation = data['bucketSampleAnnotation']
    if "fileSampleAnnotation" in data:  
        file_sample_annotation = data['fileSampleAnnotation']
    if "bucketAttributeAnnotation" in data:          
        bucket_attribute_annotation = data['bucketAttributeAnnotation']
    if "fileAttributeAnnotation" in data:                  
        file_attribute_annotation = data['fileAttributeAnnotation']

    # get name and chart title 
    name = data["name"]
    title = data["title"]             

    # get view type: sample_view (primal), attribute_view (dual)
    view = data["view"]

    # get annotation group selected: sample or attribute
    if data["filterType"] == 'filter_sample':
        annotation = data["filterSampleAnnotation"]
        filter_by = "sample_annotation"
    else:        
        annotation = data["filterAttributeAnnotation"]  
        filter_by = "attribute_annotation"

    # number of bins selected (5 by default)
    bins = None        
    if "bins" in data:
        bins = data["bins"]

    # groups and items pear grouped selected
    groups = data['groups']

    items = []
    for group in groups:
        items = items + group["values"]

    # get datamatrix dataframe from filters    
    _logger.info("Get filter items from data view %s for %s histogram", view, name)

    # get datamatrix dataframe from items selected from view=primal/dual
    #datamatrix_df = filter_datamatrix(bucket_datamatrix, file_datamatrix, view, items)

    df_datamatrix = filter_cache_datamatrix(bucket_datamatrix, file_datamatrix, view, items)
        
    # create expression dataframe
    if view == "sample_view": 
        # get sample annotations dataframe (primal)
        #df_sample_annotation = filter_annotation(bucket_sample_annotation, file_sample_annotation)
        df_sample_annotation = filter_cache_annotation(bucket_sample_annotation, file_sample_annotation)

        # get expression dataframe merging filtered datamatrix dataframe with annotation dataframe from primal view
        df_expression = pd.merge(df_datamatrix, df_sample_annotation, on=["sample_id"])        
    else:
        # get attribute annotations dataframe (dual)
        #df_attribute_annotation = filter_annotation(bucket_attribute_annotation, file_attribute_annotation)
        df_attribute_annotation = filter_cache_annotation(bucket_attribute_annotation, file_attribute_annotation)

        # get expression dataframe merging filtered datamatrix dataframe with attribute dataframe from dual view
        df_expression = pd.merge(df_datamatrix, df_attribute_annotation, on=["attribute_id"])

    # apply histogram analytics to expression dataframe
    histogram = []
    for group in groups:
        # create filtered dataframe for each group items
        items = np.array(group["values"]) 
        group_df = pd.DataFrame(data = items, columns = ["sample_id"])

        expression_grouped_df = pd.merge(df_expression, group_df, on=["sample_id"])
        
        if filter_by == "sample_annotation":
            # histogram grouped by sample annotation
            df_expression_hist = expression_grouped_df.groupby([annotation])[annotation].count()                        

            for index in df_expression_hist.index:    
                data = {}
                data["annotation"] = index
                data["group"] = group["name"]
                data["color"] = group["color"]
                data["value"] = int(df_expression_hist[index])

                histogram.append(data)            
        else:            
            # histogram grouped by attribute annotation
            expression_grouped_df["hist"]=pd.cut(expression_grouped_df[annotation], bins=bins)
            df_expression_hist = expression_grouped_df.groupby(["hist"])["hist"].count() 

            for index in df_expression_hist.index:    
                data = {}
                data["annotation"] = str(index.mid)
                data["group"] = group["name"]
                data["color"] = group["color"]
                data["value"] = int(df_expression_hist[index])

                histogram.append(data)            
        
    # order group respose by annotation
    histogram.sort(key=lambda item: item["annotation"])    

    # timestamp track
    end = time.time()
    _logger.info("Histogram processing time: " + str(end - start))

    return histogram

@app.route('/analytics/logistic_regression', methods=['POST'])
def logistic_regression():
    start = time.time()

    response = []

    # TODO
    # timestamp track
    end = time.time()
    _logger.info("Logistic Regression processing time: " + str(end - start))

    return response

def wsgi():
    global _connection_minio
    global _config

    # connect to elastic database
    _connection_minio = connect_object_storage(_config["minio"])

    return app

def main(args):
    global _client_minio
    global _config

    args, unknown = parse_args(args)
    
    setup_logging(args.loglevel)
    
    # connect to elastic database
    _client_minio = connect_object_storage(_config["minio"])

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
