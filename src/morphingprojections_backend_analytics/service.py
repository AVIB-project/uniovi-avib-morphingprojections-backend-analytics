import os
import gc
import sys
import time
import argparse
import logging
import shelve
import dbm
from io import BytesIO
from itertools import islice

from pyaml_env import parse_config

import numpy as np
import pandas as pd
import dask.dataframe as dd

from flask import Flask, jsonify, request

import boto3

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

__author__ = "Miguel Salinas Gancedo"
__copyright__ = "Miguel Salinas Gancedo"
__license__ = "MIT"

CACHE_FOLDER = "./src/morphingprojections_backend_analytics/.cache"
MAX_REGRESION_VALUES = 100

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

def get_filter_cache_datamatrix(config, bucket, key, view, items):
    # get file name from key. The bucker is the organizationId and the key has this structure: projectId/caseId/fileName
    keys = key.split('/')

    organization_id = bucket
    project_id = keys[0]
    case_id = keys[1]
    file_name = keys[2]

    # get file last modify metadata to create a unique file key to be cached
    response = _client_minio.head_object(Bucket=bucket, Key=key)
    datetime_value = response["LastModified"]
    
     # create a cache file per organization/project/case + minio last update
    cache_db = "cache_" + organization_id + "_" + project_id + "_" + case_id
    cache_file = cache_db + "_" + file_name
    file_key = cache_file + "_" + datetime_value.strftime("%Y%m%d%H%M%S")

    # Check if the file key is already in the case cache db
    with shelve.open(CACHE_FOLDER + "/" + cache_db, flag='c', writeback=False) as cache:        
        if file_key not in cache:              
            # download file from minio and cache locally
            file_stream = _client_minio.get_object(Bucket=bucket, Key=key)
            file_data = file_stream['Body'].read() 

            # parse parquet cache file to dataframe
            #df_datamatrix = pd.read_parquet(BytesIO(cache[file_key]))                
            df_datamatrix = pd.read_csv(BytesIO(file_data))                

            '''df_datamatrix = dd.read_csv(
                "s3://" + bucket + "/" + key,
                blocksize="100MB",
                storage_options={
                    "key": config["access_key"],
                    "secret": config["secret_key"],
                    "client_kwargs": {"endpoint_url": str(config["scheme"]) + "://" + str(config["host"]) + ":" + str(config["port"]), "verify": False}
                }
            )'''

            #cache[file_key] = file_data
            cache[file_key] = df_datamatrix

            _logger.info("Reading from file " + file_key)
        else:
            # If in the cache, retrieve the content from the cache
            _logger.info("Reading from cache " + file_key)
 
        # Filter datamatrix dataframe filtered by items (sample view (primal) or attribute view (dual))
        if view == "sample_view":
            # filter datamatrix dataframe rows by sample_id
            df_datamatrix = cache[file_key][cache[file_key]["sample_id"].isin(items)]            
        else:
            # filter datamatrix dataframe columns by attribute_id
            df_datamatrix = cache[file_key][items]            

            # transposed datamatrix dataframe
            df_datamatrix = df_datamatrix.T

        # Return the content either from the cache or newly read file
        return df_datamatrix

def get_filter_cache_annotation(config, bucket, key):
    # get file name from key. The bucker is the organizationId and the key has this structure: projectId/caseId/fileName
    keys = key.split('/')
    
    organization_id = bucket
    project_id = keys[0]
    case_id = keys[1]
    file_name = keys[2]

    response = _client_minio.head_object(Bucket=bucket, Key=key)
    datetime_value = response["LastModified"]

    # get file last modify metadata to create a unique file key to be cached
    cache_db = "cache_" + organization_id + "_" + project_id + "_" + case_id
    cache_file = cache_db + "_" + file_name
    file_key = cache_file + "_" + datetime_value.strftime("%Y%m%d%H%M%S")

    # Check if the file key is already in the cache
    with shelve.open(CACHE_FOLDER + "/" + cache_db) as cache:        
        if file_key not in cache:
            try: 
                # download file from minio and cache locally
                file_stream = _client_minio.get_object(Bucket=bucket, Key=key)
                file_data = file_stream['Body'].read() 

                df_annotation = pd.read_csv(BytesIO(file_data)) 

                '''df_annotation = dd.read_csv(
                    "s3://" + bucket + "/" + key,
                    blocksize="100MB",
                    storage_options={
                        "key": config["access_key"],
                        "secret": config["secret_key"],
                        "client_kwargs": {"endpoint_url": str(config["scheme"]) + "://" + str(config["host"]) + ":" + str(config["port"]), "verify": False}
                    }
                )'''

                cache[file_key] = df_annotation

                _logger.info("Reading from file " + file_key)
            except Exception as err:                
                cache.close()
                return err                
        else:
            # If in the cache, retrieve the content from the cache
            _logger.info("Reading from cache " + file_key)
 
        df_annotation = cache[file_key]

        return df_annotation

def get_filter_attributes(df_expressions):
    # return the columns (attributes) without the first one (sample_id)
    return df_expressions.drop(df_expressions.columns[0],axis=1).columns

@app.route('/')
def health():
    _logger.info("Health Endpoint")

    return jsonify('I am Ok')

@app.route('/analytics/histogram', methods=['POST'])
def histogram():
    # start tracking
    start = time.time()

    _logger.info("Get request data to execute histogram")

    # get data from request parsed
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

    # get metadata from request
    name = data["name"]
    title = data["title"]             
    view = data["view"]  # get view type: sample_view (primal), attribute_view (dual)
    groups = data['groups']

    # get annotation group selected: sample or attribute
    if data["filterType"] == 'filter_sample':
        annotation = data["filterSampleAnnotation"]
        filter_by = "sample_annotation"
    else:        
        annotation = data["filterAttributeAnnotation"]  
        filter_by = "attribute_annotation"

    # get number of bins selected (5 by default)
    bins = None        
    if "bins" in data:
        bins = data["bins"]

    # aggregate all items from groups
    items = []
    for group in groups:
        items = items + group["values"]

    _logger.info("Get and cache Datamatrix from view %s and name %s", view, name)
    df_expression = get_filter_cache_datamatrix(_config["minio"], bucket_datamatrix, file_datamatrix, view, items)
        
    # create expression dataframe from cache datamatrix and annotations cache file    
    if view == "sample_view": 
        # get sample annotations dataframe (primal)
        _logger.info("Get and cache Sample Annotations")
        df_sample_annotation = get_filter_cache_annotation(_config["minio"], bucket_sample_annotation, file_sample_annotation)
        
        # get expression dataframe merging filtered datamatrix dataframe with annotation dataframe from primal view
        _logger.info("Merge Datamatrix with Sample Annotations")
        df_expression = df_expression.merge(df_sample_annotation, how='inner', on='sample_id')
    else:
        # get attribute annotations dataframe (dual)
        _logger.info("Get and cache Attribute Annotations")
        df_attribute_annotation = get_filter_cache_annotation(_config["minio"], bucket_attribute_annotation, file_attribute_annotation)

        # get expression dataframe merging filtered datamatrix dataframe with attribute dataframe from dual view
        _logger.info("Merge Datamatrix with Attribute Annotations")
        df_expression = df_expression.merge(df_attribute_annotation, how='inner', on='attribute_id')

    # apply histogram analytics to expression dataframe
    lst_histogram = []
    for group in groups:
        # create filtered dataframe for each group items
        items = np.array(group["values"]) 
        df_group = pd.DataFrame(data = items, columns = ["sample_id"])
        #df_group = dd.from_array(items, columns=['sample_id'])

        # add sample annotation metadata to groups
        df_expression_grouped = df_expression.merge(df_group, on=["sample_id"])
        
        # create histogram
        if filter_by == "sample_annotation":
            # histogram grouped by sample annotation
            df_expression_hist = df_expression_grouped.groupby([annotation])[annotation].count()                        

            for index in df_expression_hist.index:    
                data = {}
                data["annotation"] = index
                data["group"] = group["name"]
                data["color"] = group["color"]
                data["value"] = int(df_expression_hist[index])
                #data["value"] = int(df_expression_hist[index].compute().values)

                lst_histogram.append(data)            
        else:            
            # histogram grouped by attribute annotation
            df_expression_grouped["hist"]=pd.cut(df_expression_grouped[annotation], bins=bins)
            df_expression_hist = df_expression_grouped.groupby(["hist"])["hist"].count() 

            for index in df_expression_hist.index:    
                data = {}
                data["annotation"] = str(index.mid)
                data["group"] = group["name"]
                data["color"] = group["color"]
                data["value"] = int(df_expression_hist[index])

                lst_histogram.append(data)            
        
    # order histogram by annotation
    lst_histogram.sort(key=lambda item: item["annotation"])    

    # timestamp track
    end = time.time()
    _logger.info("Histogram processing time: " + str(end - start))

    return lst_histogram

@app.route('/analytics/logistic_regression', methods=['POST'])
def logistic_regression():
    # start tracking
    start = time.time()

    _logger.info("Get request data to execute logistic regression")

    # get data from request parsed
    data = request.get_json() 

    # get resource files from request
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

    # get metadata from request
    name = data["name"]
    title = data["title"] 
    view = data["view"] # get view type: sample_view (primal), attribute_view (dual)
    groups = data['groups']

    # get annotations filters from request
    sample_annotation = None
    if "filterSampleAnnotation" in data:   
        sample_annotation = data["filterSampleAnnotation"]

    attribute_annotation = None
    if "filterAttributeAnnotation" in data:           
        attribute_annotation = data["filterAttributeAnnotation"] 

    # aggregate all items from groups
    items = []
    for index, group in  enumerate(groups):
        items = items + group["values"]

    sample_group_lst = []
    for index_group, group in enumerate(groups):
        for sample_id in group["values"]:
            sample_group_lst.append({'sample_id': sample_id, 'group_id': index_group})        

    # get expressions from db
    _logger.info("Get and cache datamatrix from view %s and name %s to execute Logistic Regression", view, name)
    df_expressions = get_filter_cache_datamatrix(_config["minio"], bucket_datamatrix, file_datamatrix, view, items)

    # parse group collection to dataframe
    sample_groups = pd.json_normalize(sample_group_lst)

    # merge expression with group dataframes
    df_expressions = df_expressions.merge(sample_groups, how='inner', on='sample_id')
    df_expressions = df_expressions.sort_values('group_id', ascending=True)

    # get logistic regression subdataframes to be trained
    X = df_expressions.iloc[:,1:-1]
    y = df_expressions.loc[:,"group_id"]

    LR = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(X, y)

    # get logistic regression coeficients (normal vector hyperplane)
    d = LR.coef_
    
    # normalize logistic regression results
    d = np.abs(d) # set absolute values
    d = d/np.max(np.max(d)) # Normalize dataframe
    d = d[0]
    
    response = []

    # sort results
    d = sorted(d, reverse=True)

    # trunc and get the first 100 values
    d = islice(d, MAX_REGRESION_VALUES)

    for index_d, regression in enumerate(d):
        df_sub_expression = df_expressions[[df_expressions.columns[index_d + 1], "group_id"]]
        df_mean_expressions = df_sub_expression.groupby(by=["group_id"]).mean()
        df_standard_deviation_expressions = df_sub_expression.groupby(by=["group_id"]).std()
        
        analytics_a = str(round(df_mean_expressions.values[0][0], 2)) + "±" + str(round(df_standard_deviation_expressions.values[0][0], 2))
        analytics_b = str(round(df_mean_expressions.values[1][0], 2)) + "±" + str(round(df_standard_deviation_expressions.values[1][0], 2))
        #analytics_a = str(round(df_mean_expressions.compute().values[0][0], 2)) + "±" + str(round(df_standard_deviation_expressions.compute().values[0][0], 2))
        #analytics_b = str(round(df_mean_expressions.compute().values[1][0], 2)) + "±" + str(round(df_standard_deviation_expressions.compute().values[1][0], 2))

        response.append(
            {
                "attribute": df_expressions.columns[index_d + 1], 
                "analytics_a": analytics_a,
                "analytics_b": analytics_b,
                "value": regression
            }) 

    #response.sort(key=lambda x: x["value"], reverse=True)

    # end tracking
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
