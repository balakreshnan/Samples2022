# Automated ML Batch score custom code

## Use Automated ML model to batch score large volume data

## Prerequisites

- Azure account
- Azure ML account
- Azure Storage
- Create a folder for Training data, Batch score input, Batch score output
- Create a compute cluster
- Need notebook or jupyter lab or jupyter to write code

## Code

- Load the existing model got registered by automated ML
- Go to model section and pull the model name

```
from azureml.core.model import Model
model = Model(ws, 'AutoMLxxxxxxxx')
```

- download the model

```
model.download(target_dir='./titanic3model', exist_ok=False, exists_ok=None)
```

- Now import necessary libraries

```
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
```

- now load the model

```
model = joblib.load('./titanic3model/model.pkl')
```

- NOw lets load the new batch score data
- I have already registered the data store and dataset

```
dataset_name = 'titanicbatchscoreinput'

# Get a dataset by name
scoretitanic_ds = Dataset.get_by_name(workspace=ws, name=dataset_name)

# Load a TabularDataset into pandas DataFrame
dfscoreinput = scoretitanic_ds.to_pandas_dataframe()
```

- drop prediction column

```
dfscoreinput1 = dfscoreinput.drop('survived', 1)
```

- predict the output

```
result1 = model.predict(dfscoreinput1)
```

- Append the predicted output and add to existing dataset to send

```
dfscoreinput['PredictedSurvived'] = result1
```

- save the file to output

```
dfscoreinput.to_csv('output3/titanic3output.csv')
```

- upload the new predicted output to adls gen2

```
from azureml.core import Workspace
ws = Workspace.from_config()
#datastore = ws.get_default_datastore()
datastore = Datastore.get(ws, 'titanicbatchscoreoutput')
    
datastore.upload(src_dir='./output3',
                  target_path='output/',
                  overwrite=True)
```

## done