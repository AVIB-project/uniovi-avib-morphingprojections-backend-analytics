import os
import sys
import time
import argparse
import logging
import shelve
from io import BytesIO

from pyaml_env import parse_config

import numpy as np
import pandas as pd

from flask import Flask, jsonify, request

import boto3

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

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

def filter_cache_datamatrix(bucket, key, view, items):
    # get file name from key. The bucker is the organizationId and the key has this structure: projectId/caseId/fileName
    keys = key.split('/')

    project_id = keys[0]
    case_id = keys[1]
    file_name = keys[2]

    # get file last modify metadata to create a unique file key to be cached
    response = _client_minio.head_object(Bucket=bucket, Key=key)
    datetime_value = response["LastModified"]
     # create a cache db per organization
    #cache_db = "cache_" + bucket + ".db"
    cache_file = "cache_" + bucket + "_" + project_id + "_" + case_id + "_" + file_name
    file_key = cache_file + "_" + datetime_value.strftime("%Y%m%d%H%M%S")

    # Check if the file key is already in the cache
    with shelve.open(CACHE_FOLDER + "/" + cache_file) as cache:        
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
        #df_datamatrix = pd.read_parquet(BytesIO(cache[file_key]))
        df_datamatrix = pd.read_csv(BytesIO(cache[file_key]))

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

def filter_cache_annotation(bucket, key):
    # get file name from key. The bucker is the organizationId and the key has this structure: projectId/caseId/fileName
    keys = key.split('/')
    
    project_id = keys[0]
    case_id = keys[1]
    file_name = keys[2]

    # get file last modify metadata to create a unique file key to be cached
    response = _client_minio.head_object(Bucket=bucket, Key=key)
    datetime_value = response["LastModified"]
    file_key = file_name + "_" + datetime_value.strftime("%Y%m%d%H%M%S")

     # create a cache db per organization
    #cache_db = "cache_" + bucket + ".db"
    cache_file = "cache_" + bucket + "_" + project_id + "_" + case_id + "_" + file_name

    # Check if the file key is already in the cache
    with shelve.open(CACHE_FOLDER + "/" + cache_file) as cache:        
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
        #df_annotation = pd.read_parquet(BytesIO(cache[file_key]))
        df_annotation = pd.read_csv(BytesIO(cache[file_key]))

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

    # get datamatrix dataframe from items selected from view=primal/dual
    _logger.info("Cache Datamatrix for view %s and %s histogram", view, name)

    #datamatrix_df = filter_datamatrix(bucket_datamatrix, file_datamatrix, view, items)
    df_datamatrix = filter_cache_datamatrix(bucket_datamatrix, file_datamatrix, view, items)
        
    # create expression dataframe from cache datamatrix and annotations cache file
    _logger.info("Prepare expression datamatrix for view %s and %s histogram", view, name)
    if view == "sample_view": 
        _logger.info("Cache Sample Annotations for view %s and %s histogram", view, name)

        # get sample annotations dataframe (primal)
        #df_sample_annotation = filter_annotation(bucket_sample_annotation, file_sample_annotation)
        df_sample_annotation = filter_cache_annotation(bucket_sample_annotation, file_sample_annotation)

        _logger.info("Merge Datamatrix with Sample Annotations for view %s and %s histogram", view, name)

        # get expression dataframe merging filtered datamatrix dataframe with annotation dataframe from primal view
        df_expression = pd.merge(df_datamatrix, df_sample_annotation, on=["sample_id"])        
    else:
        _logger.info("Cache Attribute Annotations file for view %s and %s histogram", view, name)

        # get attribute annotations dataframe (dual)
        #df_attribute_annotation = filter_annotation(bucket_attribute_annotation, file_attribute_annotation)
        df_attribute_annotation = filter_cache_annotation(bucket_attribute_annotation, file_attribute_annotation)

        _logger.info("Merge Datamatrix with Attribute Annotations for view %s and %s histogram", view, name)

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

    end = time.time()
    _logger.info("Logistic Regression processing time: " + str(end - start))

    return response

def wsgi():
    global _client_minio
    global _config

    # connect to elastic database
    _client_minio = connect_object_storage(_config["minio"])

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
