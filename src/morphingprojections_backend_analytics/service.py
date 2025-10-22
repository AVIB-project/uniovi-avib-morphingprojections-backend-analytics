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

import pyarrow.parquet as pq

from minio import Minio
from minio.error import MinioException

from sklearn.linear_model import LogisticRegression

from flask import Flask, jsonify, request

__author__ = "Miguel Salinas Gancedo"
__copyright__ = "Miguel Salinas Gancedo"
__license__ = "MIT"

_CACHE_FOLDER = "./src/morphingprojections_backend_analytics/.cache"
_MAX_REGRESION_VALUES = 100

_logger = logging.getLogger(__name__)

# get environment variables from active profile            
if not os.getenv('PYTHON_PROFILES_ACTIVE'):
    _config = parse_config('./src/morphingprojections_backend_analytics/environment/environment.yaml')        
else:
    _config = parse_config('./src/morphingprojections_backend_analytics/environment/environment-' + os.getenv('PYTHON_PROFILES_ACTIVE') + '.yaml')

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
    parser = argparse.ArgumentParser(description="Analytics Backend Service")
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
    return Minio(str(config["host"]) + ":" + str(config["port"]),
        access_key=config["access_key"],
        secret_key=config["secret_key"],
        cert_check=False)        

def get_filter_cache_datamatrix(config, bucket, key, view, items):
    # get file name from key. The bucker is the organizationId and the key has this structure: projectId/caseId/fileName
    key_tokens = key.split('/')

    organization_id = bucket
    project_id = key_tokens[0]
    case_id = key_tokens[1]
    file_name = key_tokens[2]

    file_name_tokens = file_name.split('.')
    file_name = file_name_tokens[0] + ".parquet"
    key = project_id + "/" + case_id + "/" + file_name

    # get file last modify metadata to create a unique file key to be cached
    response = _client_minio.stat_object(bucket_name=bucket, object_name=key)    
    datetime_value = response.last_modified

     # create a cache file per organization/project/case + minio last update
    cache_db = "cache_" + organization_id + "_" + project_id + "_" + case_id
    cache_file = cache_db + "_" + file_name
    file_key = cache_file + "_" + datetime_value.strftime("%Y%m%d%H%M%S")

    # Check if the file key is already in the case cache db
    with shelve.open(_CACHE_FOLDER + "/" + cache_db, flag='c', writeback=False) as cache:        
        if file_key not in cache:              
            # download file from minio and cache locally
            file_stream = _client_minio.get_object(bucket_name=bucket, object_name=key)

            #df_datamatrix = pd.read_csv(BytesIO(file_stream.read()))

            table = pq.read_table(BytesIO(file_stream.read()))
            df_datamatrix = table.to_pandas()

            '''
            df_datamatrix = dd.read_csv(
                "s3://" + bucket + "/" + key,
                blocksize="100MB",
                storage_options={
                    "key": config["access_key"],
                    "secret": config["secret_key"],
                    "client_kwargs": {"endpoint_url": str(config["scheme"]) + "://" + str(config["host"]) + ":" + str(config["port"]), "verify": False}
                }
            )
            '''          

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

    # get file last modify metadata to create a unique file key to be cached
    response = _client_minio.stat_object(bucket_name=bucket, object_name=key)    
    datetime_value = response.last_modified

    # get file last modify metadata to create a unique file key to be cached
    cache_db = "cache_" + organization_id + "_" + project_id + "_" + case_id
    cache_file = cache_db + "_" + file_name
    file_key = cache_file + "_" + datetime_value.strftime("%Y%m%d%H%M%S")

    # Check if the file key is already in the cache
    with shelve.open(_CACHE_FOLDER + "/" + cache_db) as cache:        
        if file_key not in cache:
            # download file from minio and cache locally
            file_stream = _client_minio.get_object(bucket_name=bucket, object_name=key)

            df_annotation = pd.read_csv(BytesIO(file_stream.read()))
            #df_datamatrix = pd.read_parquet(BytesIO(file_stream.read()))

            '''
            df_annotation = dd.read_csv(
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
        else:
            # If in the cache, retrieve the content from the cache
            _logger.info("Reading from cache " + file_key)
 
        df_annotation = cache[file_key]

        return df_annotation

@app.route('/')
def health():
    _logger.info("Health Endpoint")

    return jsonify('I am Ok')

@app.route('/analytics/histogram', methods=['POST'])
def histogram():
    # start tracking
    start = time.time()
    _logger.info("Start Histogram processing at time: " + str(start))

    # parse request
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

    # get datamatrix from cache filtered  by items
    _logger.info("Get and cache Datamatrix filtered by items from view %s and name %s", view, name)
    df_expression = get_filter_cache_datamatrix(_config["minio"], bucket_datamatrix, file_datamatrix, view, items)
        
    # create dataframe merged with annotations   
    if view == "sample_view":
        if filter_by == "sample_annotation":
            _logger.info("Get and cache Sample Annotations")
            df_sample_annotation = get_filter_cache_annotation(_config["minio"], bucket_sample_annotation, file_sample_annotation)
        
            _logger.info("Merge Datamatrix with Sample Annotations")
            df_expression = df_expression.merge(df_sample_annotation, how='inner', on='sample_id')
        else:
            _logger.info("Filte by Attribute Annotation")
            df_expression = df_expression.loc[:,["sample_id", annotation]]
    else:
        if filter_by == "sample_annotation":
            _logger.info("Get and cache Attribute Annotations")
            df_attribute_annotation = get_filter_cache_annotation(_config["minio"], bucket_attribute_annotation, file_attribute_annotation)

            _logger.info("Merge Datamatrix with Attribute Annotations")
            df_expression = df_expression.merge(df_attribute_annotation, how='inner', on='attribute_id')
        else:
            _logger.info("Filte by Sample Annotation")
            df_expression = df_expression.loc[:,["attribute_id", annotation]]

    # apply histogram analytics to expression dataframe
    df_expression_graph = pd.DataFrame()
    for group in groups:
        # get items selected by group       
        items = np.array(group["values"])

        # filter datamatrix with selected items by group       
        if view == "sample_view":
            df_expression_filtered_by_group = df_expression[df_expression["sample_id"].isin(items)] 
        else:     
            df_expression_filtered_by_group = df_expression[df_expression["attribute_id"].isin(items)]           
        
        # datamatrix grouped by annotarion
        if filter_by == "sample_annotation":
            df_expression_hist = df_expression_filtered_by_group.groupby(annotation).size().reset_index(name='value')
            df_expression_hist["group"] = group["name"]
            df_expression_hist["color"] = group["color"]        
            df_expression_hist.rename(columns={annotation: "annotation"}, inplace=True) 

            # concatenate all group datamatrix                       
            df_expression_graph = pd.concat([df_expression_hist, df_expression_graph], ignore_index=True)            
        else:     
            df_expression_filtered_by_group['annotation'] = pd.cut(df_expression_filtered_by_group[annotation], bins=bins)
            df_expression_hist = df_expression_filtered_by_group.groupby('annotation').count()
            df_expression_hist = df_expression_hist.reset_index()            
            df_expression_hist["annotation"] = df_expression_hist["annotation"].astype('string')
            df_expression_hist = df_expression_hist.drop(['sample_id'], axis=1)
            df_expression_hist.rename(columns={annotation: "value"}, inplace=True)
            df_expression_hist["group"] = group["name"]
            df_expression_hist["color"] = group["color"]            

            # concatenate all group datamatrix                       
            df_expression_graph = pd.concat([df_expression_hist, df_expression_graph])

    # format group datamatrix
    df_expression_graph = df_expression_graph.pivot(index=['group', 'color'], columns='annotation', values='value').fillna(0)
    lst_histogram = df_expression_graph.reset_index().to_dict(orient='records')

    # end histogram
    _logger.info("End Histogram processing at time: " + str(time.time() - start))

    return lst_histogram

@app.route('/analytics/logistic_regression', methods=['POST'])
def logistic_regression():
    # start tracking
    start = time.time()
    _logger.info("Start Logistic Regression processing at time: " + str(start))

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

    # aggregate all items from groups
    items = []
    for group in groups:
        items = items + group["values"]

    sample_group_lst = []
    for index_group, group in enumerate(groups):
        for sample_id in group["values"]:
            sample_group_lst.append({'sample_id': sample_id, 'group_id': index_group})        

    # get datamatrix from cache filtered  by items
    _logger.info("Get and cache Datamatrix filtered by items from view %s and name %s", view, name)
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
    d = d/np.max(d) # Normalize dataframe
    d = d[0]
    
    response = []
    for index_d, regression in enumerate(d):
        df_sub_expression = df_expressions[[df_expressions.columns[index_d + 1], "group_id"]]
        df_mean_expressions = df_sub_expression.groupby(by=["group_id"]).mean()
        df_standard_deviation_expressions = df_sub_expression.groupby(by=["group_id"]).std()
        
        analytics_a = str(round(df_mean_expressions.values[0][0], 2)) + "±" + str(round(df_standard_deviation_expressions.values[0][0], 2))
        analytics_b = str(round(df_mean_expressions.values[1][0], 2)) + "±" + str(round(df_standard_deviation_expressions.values[1][0], 2))

        response.append(
            {
                "attribute": df_expressions.columns[index_d + 1], 
                "analytics_a": analytics_a,
                "analytics_b": analytics_b,
                "value": regression
            }) 

    response.sort(key=lambda x: x["value"], reverse=True)

    response = response[:_MAX_REGRESION_VALUES]

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
