# Build classifiction model using titanic data Train, Test, Explanation, Model Interpretation, Deploy and Model Drift using AutoML

## Using automated ML to build a model to deploy end to end

## Steps

- Use Open source Titanic data to build a model
- Use AutoML to build a model
- Train, Register, Deploy the model using AKS
- Test the rest service

## Prerequistie

## Code

```
import logging

from matplotlib import pyplot as plt
import pandas as pd

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset
from azureml.train.automl import AutoMLConfig
from azureml.interpret import ExplanationClient
```

```
from azureml.core import Workspace
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.model import Model
```

```
import azureml.core
print(azureml.core.VERSION)
```

```
from azureml.core.experiment import Experiment

ws = Workspace.from_config()
```

- create a automatedml config

```
from azureml.train.automl import AutoMLConfig

# task can be one of classification, regression, forecasting
automl_config = AutoMLConfig(task = "classification")
```

- Get data set for training

```
from azureml.core.dataset import Dataset
data = "https://dprepdata.blob.core.windows.net/demo/Titanic.csv"
dataset = Dataset.Tabular.from_delimited_files(data)
```

- Set the label column name field

```
label = "Survived"
```

- Now create validation dataset

```
validation_data = "https://dprepdata.blob.core.windows.net/demo/Titanic.csv"
validation_dataset = Dataset.Tabular.from_delimited_files(validation_data)
```

- Now create Test data set

```
test_data = "https://dprepdata.blob.core.windows.net/demo/Titanic.csv"
test_dataset = Dataset.Tabular.from_delimited_files(test_data)
```

- create a compute if needed

```
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your CPU cluster
cpu_cluster_name = "cpu-cluster"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                           max_nodes=6)
    compute_target = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)
```

- Now configure the AutoMLConfig

```
automl_settings = {
    "experiment_timeout_hours" : 0.3,
    "enable_early_stopping" : True,
    "iteration_timeout_minutes": 5,
    "max_concurrent_iterations": 4,
    "max_cores_per_iteration": -1,
    #"n_cross_validations": 2,
    "primary_metric": 'AUC_weighted',
    "featurization": 'auto',
    "verbosity": logging.INFO,
}

automl_config = AutoMLConfig(task = 'classification',
                             debug_log = 'automl_errors.log',
                             compute_target=compute_target,
                             experiment_exit_score = 0.9984,
                             blocked_models = ['KNN','LinearSVM'],
                             enable_onnx_compatible_models=True,
                             training_data = titanic_ds,
                             label_column_name = label,
                             validation_data = validation_dataset,
                             **automl_settings
                            )
```

- Create an experiment

```
from azureml.core.experiment import Experiment
import azureml.core.runconfig
from azureml.core import Environment, Experiment
from azureml.core import ScriptRunConfig
from azureml.widgets import RunDetails

ws = Workspace.from_config()

# Choose a name for the experiment and specify the project folder.
experiment_name = 'Titanic-automl'
project_folder = './titanic/automl-classification'

experiment = Experiment(ws, experiment_name)
```

- Submit the experiment

```
run = experiment.submit(automl_config, show_output=True)
```

```
from azureml.widgets import RunDetails
RunDetails(run).show()
```

```
run.wait_for_completion()
```

```
best_run, fitted_model = run.get_output()
print(best_run)
print(fitted_model)
```

- Explain the model

```
import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

from azureml.core import Run, Dataset, Workspace, Experiment

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score,roc_curve
from interpret.ext.blackbox import TabularExplainer
from azureml.interpret import ExplanationClient
```

- Download the data

```
model_name = best_run.properties['model_name']

script_file_name = 'inference/score.py'

best_run.download_file('outputs/scoring_file_v_1_0_0.py', 'inference/score.py')
```

- Update the score.py
- Add application insights value
- We are saving probability of the model in application insights to plot model drift

```
# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
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

input_sample = pd.DataFrame({"PassengerId": pd.Series([0], dtype="int64"), "Pclass": pd.Series([0], dtype="int64"), "Name": pd.Series(["example_value"], dtype="object"), "Sex": pd.Series(["example_value"], dtype="object"), "Age": pd.Series([0.0], dtype="float64"), "SibSp": pd.Series([0], dtype="int64"), "Parch": pd.Series([0], dtype="int64"), "Ticket": pd.Series(["example_value"], dtype="object"), "Fare": pd.Series([0.0], dtype="float64"), "Cabin": pd.Series(["example_value"], dtype="object"), "Embarked": pd.Series(["example_value"], dtype="object")})
output_sample = np.array([False])
method_sample = StandardPythonParameterType("predict")

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('method', method_sample, convert_to_provided_type=False)
@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data, method="predict_proba"):
    try:
        if method == "predict_proba":
            result = model.predict_proba(data)
        elif method == "predict":
            result = model.predict(data)
        else:
            raise Exception(f"Invalid predict method argument received ({method})")
        if isinstance(result, pd.DataFrame):
            result = result.values
            
        #info = {
        #    "input": data,
        #    "output": result.tolist()
        #    }
        #print(json.dumps(info))
        logger.info("results:", result.tolist())
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
```

- register the model

```
description = 'AutoML Model trained on bank marketing data to predict if a client will subscribe to a term deposit'
tags = None
model = run.register_model(model_name = model_name, description = description, tags = tags)

print(run.model_id) # This will be written to the script file later in the notebook.
```

```
print(model.name, model.description, model.version)
```

- Now it's time to create new endpoint

```
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your AKS cluster
aks_name = 'aksrest' 

# Verify that cluster does not exist already
try:
    aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # Use the default configuration (can also provide parameters to customize)
    prov_config = AksCompute.provisioning_configuration()

    # Create the cluster
    aks_target = ComputeTarget.create(workspace = ws, 
                                    name = aks_name, 
                                    provisioning_configuration = prov_config)

if aks_target.get_status() != "Succeeded":
    aks_target.wait_for_completion(show_output=True)
```

- Environment is important

```
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies 

conda_deps = CondaDependencies.create(conda_packages=['numpy','scikit-learn','scipy'], pip_packages=['azureml-defaults', 'inference-schema', 'pandas', 'scikit-learn', 'joblib', 'azureml-train-automl-runtime','inference-schema','azureml-interpret','azureml-defaults'])
myenv = Environment(name='myenv')
myenv.python.conda_dependencies = conda_deps
```

```
from azureml.core import Workspace
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.model import Model
```

```
from azureml.core.model import InferenceConfig

inf_config = InferenceConfig(entry_script='./inference/score.py', environment=myenv)
```

```
# Set the web service configuration (using default here)
aks_config = AksWebservice.deploy_configuration()

# # Enable token auth and disable (key) auth on the webservice
# aks_config = AksWebservice.deploy_configuration(token_auth_enabled=True, auth_enabled=False)
```

- delete endpoint if exists

```
aks_service.delete()
```

```
%%time
aks_service_name ='titanicrestaks'

aks_service = Model.deploy(workspace=ws,
                           name=aks_service_name,
                           models=[model],
                           inference_config=inf_config,
                           deployment_config=aks_config,
                           deployment_target=aks_target)

aks_service.wait_for_deployment(show_output = True)
print(aks_service.state)
```

- enable application insights for api

```
aks_service.update(enable_app_insights=True)
```

```
%%time
import json
#1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
#2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
#'method' : 'predict_proba'

test_sample = json.dumps({'data': [
    [1,3,"Braund, Mr. Owen Harris","male",22,1,0,"A/5 21171",7.25,"","S"], 
    [2,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)","female",38,1,0,"PC 17599",71.2833,"C85","C"]
]})
test_sample = bytes(test_sample,encoding = 'utf8')
print(test_sample)

prediction = aks_service.run(input_data = test_sample)
print(prediction)
```

```
%%time
aks_service.delete()
model.delete()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automl1.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automl2.jpg "Architecture")