# Azure ML Designer consuming mongodb

## Use MongoDB client in Azure ML Designer

## Prerequisites

- Azure Acoount
- Storage account
- Azure Machine learning
- Compute instance
- Install Mongodb community version in compute instance using terminal
- https://www.mongodb.com/docs/manual/introduction/
- I used Ubuntu 18.04

## Code

- Now create a new Designer experiment
- Drag sample dataset and drop in canvas
- Now drag and drop Execute Python Script
- This example shows how to use mongodb client in Azure ML Designer

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/mongo1.jpg "Architecture")

- Here is the code

```

# The script MUST contain a function named azureml_main
# which is the entry point for this module.
# imports up here can be used to
import pandas as pd
import os
os.system(f"pip install pymongo")
# The entry point function MUST have two input arguments.
# If the input port is not connected, the corresponding
# dataframe argument will be None.
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1 = None, dataframe2 = None):

    # Execution logic goes here
    print(f'Input pandas.DataFrame #1: {dataframe1}')
    import importlib.util
    package_name = 'pymongo'
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        import os
        os.system(f"pip install pymongo")

    # If a zip file is connected to the third input port,
    # it is unzipped under "./Script Bundle". This directory is added
    # to sys.path. Therefore, if your zip file contains a Python file
    # mymodule.py you can import it using:
    # import mymodule
    from pymongo import MongoClient
    myclient = MongoClient("mongodb://localhost:27017")
    #myclient = MongoClient(staging_mongo_conn)
    DB_NAME = 'GL_PRED_JUBILANT_HISTORICAL'
    db = myclient.get_database(DB_NAME)
    col_name = 'IND_DUMP'
    coll = db[col_name]
    print(coll)
    x = coll.find_one()
    print(x)

    # Return value must be of a sequence of pandas.DataFrame
    # E.g.
    #   -  Single return value: return dataframe1,
    #   -  Two return values: return dataframe1, dataframe2
    return dataframe1,
```

- Save and Submit the experiment
- Wait for the Run to complete