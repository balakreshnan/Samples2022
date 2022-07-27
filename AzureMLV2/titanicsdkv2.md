# Azure Machine Learning SDK v2 using AutoML

## Use Azure Machine Learning SDK v2 to train a model, Register and Deploy a model

## Prerequisites

- Azure Account
- Azure Machine Learning workspace
- Azure Storage Account
- Get titanic.csv
- Create folders
    - create a data folder
    - Training folder
    - Test folder
    - Validate folder
- Upload the files to the above folders

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/titsdkv2-1.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/titsdkv2-2.jpg "Architecture")

## Code

- Import libraries

```
# Import required libraries
from azure.identity import DefaultAzureCredential
from azure.identity import AzureCliCredential
from azure.ai.ml import automl, Input, MLClient

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.automl import (
    classification,
    ClassificationPrimaryMetrics,
    ClassificationModels,
)
```

- Load workspace

```
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

credential = DefaultAzureCredential()
ml_client = None
try:
    ml_client = MLClient.from_config(credential)
except Exception as ex:
    print(ex)
    # Enter details of your AzureML workspace
    subscription_id = "xxxxx-xxxxxx-xxxxxx"
    resource_group = "rgname"
    workspace = "workspacename"
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)
```

- Display the workspace

```
workspace = ml_client.workspaces.get(name=ml_client.workspace_name)

subscription_id = ml_client.connections._subscription_id
resource_group = workspace.resource_group
workspace_name = ml_client.workspace_name

output = {}
output["Workspace"] = workspace_name
output["Subscription ID"] = subscription_id
output["Resource Group"] = resource_group
output["Location"] = workspace.location
output
```

- Configure training data

```
# Create MLTables for training dataset

my_training_data_input = Input(
    type=AssetTypes.MLTABLE, path="./data/training-mltable-folder"
)
```

- Configure experiment name

```
# General job parameters
compute_name = "cpu-cluster"
max_trials = 5
exp_name = "automlv2-Titanic-classifier-experiment"
```

- Automl Configuration

```
# Create the AutoML classification job with the related factory-function.

classification_job = automl.classification(
    compute=compute_name,
    experiment_name=exp_name,
    training_data=my_training_data_input,
    target_column_name="Survived",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True,
    tags={"my_custom_tag": "My Titanic Automl Exp"},
)

# Limits are all optional
classification_job.set_limits(
    timeout_minutes=600,
    trial_timeout_minutes=20,
    max_trials=max_trials,
    # max_concurrent_trials = 4,
    # max_cores_per_trial: -1,
    enable_early_termination=True,
)

# Training properties are optional
classification_job.set_training(
    blocked_training_algorithms=[ClassificationModels.LOGISTIC_REGRESSION],
    enable_onnx_compatible_models=True,
)
```

- Create a job

```
# Submit the AutoML job
returned_job = ml_client.jobs.create_or_update(
    classification_job
)  # submit the job to the backend

print(f"Created job: {returned_job}")
```

- Get the details

```
ml_client.jobs.stream(returned_job.name)
```

- Get the endpoint url

```
# Get a URL for the status of the job
returned_job.services["Studio"].endpoint
print(returned_job.name)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/titsdkv2-3.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/titsdkv2-4.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/titsdkv2-5.jpg "Architecture")

- Display best model

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/titsdkv2-6.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/titsdkv2-7.jpg "Architecture")

- Invoke MLFLow to get details

```
import mlflow

# Obtain the tracking URL from MLClient
MLFLOW_TRACKING_URI = ml_client.workspaces.get(
    name=ml_client.workspace_name
).mlflow_tracking_uri

print(MLFLOW_TRACKING_URI)
```

```
# Set the MLFLOW TRACKING URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))
```

- Enabling MLFLow

```
from mlflow.tracking.client import MlflowClient

# Initialize MLFlow client
mlflow_client = MlflowClient()
```

```
job_name = returned_job.name

# Example if providing an specific Job name/ID
# job_name = "b4e95546-0aa1-448e-9ad6-002e3207b4fc"

# Get the parent run
mlflow_parent_run = mlflow_client.get_run(job_name)

print("Parent Run: ")
print(mlflow_parent_run)
```

- get tags

```
# Print parent run tags. 'automl_best_child_run_id' tag should be there.
print(mlflow_parent_run.data.tags)
```

- Get the best run

```
# Get the best model's child run

best_child_run_id = mlflow_parent_run.data.tags["automl_best_child_run_id"]
print("Found best child run id: ", best_child_run_id)

best_run = mlflow_client.get_run(best_child_run_id)

print("Best child run: ")
print(best_run)
```

- display the metrics

```
best_run.data.metrics
```

- Create folder to download machine learning

```
import os

# Create local folder
local_dir = "./artifact_downloads"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)
```

```
# Download run's artifacts/outputs
local_path = mlflow_client.download_artifacts(
    best_run.info.run_id, "outputs", local_dir
)
print("Artifacts downloaded in: {}".format(local_path))
print("Artifacts: {}".format(os.listdir(local_path)))
```

```
# Show the contents of the MLFlow model folder
os.listdir("./artifact_downloads/outputs/mlflow-model")
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/titsdkv2-8.jpg "Architecture")

- Now create Managed online

```
# import required libraries
# import required libraries
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
    ProbeSettings,
)
from azure.ai.ml.constants import ModelType
```

```
model_name = "titanic-model-v2"
model = Model(
    path=f"azureml://jobs/{best_run.info.run_id}/outputs/artifacts/outputs/model.pkl",
    name=model_name,
    description="my sample titanic mlflow model",
)

# for downloaded file
# model = Model(path="artifact_downloads/outputs/model.pkl", name=model_name)

registered_model = ml_client.models.create_or_update(model)
```

```
registered_model.id
```

```
# Creating a unique endpoint name with current datetime to avoid conflicts
import datetime

online_endpoint_name = "titanic-" + datetime.datetime.now().strftime("%m%d%H%M%f")

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="this is a sample online endpoint for titanic mlflow model",
    auth_mode="key",
    tags={"foo": "bar"},
)
```

```
ml_client.begin_create_or_update(endpoint)
```

- Setup environment

```
env = Environment(
    name="automl-titanic-tabular-env",
    description="environment for automl inference",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210727.v1",
    conda_file="artifact_downloads/outputs/conda_env_v_1_0_0.yml",
)
```

- bring the scoring script to use for managed endpoint api

```
code_configuration = CodeConfiguration(
    code="artifact_downloads/outputs/", scoring_script="scoring_file_v_2_0_0.py"
)
```

- Setup the deployment

```
deployment = ManagedOnlineDeployment(
    name="titanic-deploy",
    endpoint_name=online_endpoint_name,
    model=registered_model.id,
    environment=env,
    code_configuration=code_configuration,
    instance_type="Standard_DS2_V2",
    instance_count=1,
)
```

- update the deployment for above endpoint

```
ml_client.online_deployments.begin_create_or_update(deployment)
```

- Set the traffic 

```
# bankmarketing deployment to take 100% traffic
endpoint.traffic = {"titanic-deploy": 100}
ml_client.begin_create_or_update(endpoint)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/titsdkv2-9.jpg "Architecture")

- Details of endpoint

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/titsdkv2-10.jpg "Architecture")

- Test the endpoint
- Configure sample data

```
{"data": [
    [ 1, "3", "Braund Mr. Owen Harris", "male", "22", "1", "0", "A/5 21171", 7.25, "C76", "S"],
    [ 2, "1", "Cumings Mrs. John Bradley", "female", "38", "1", "0", "PC 17599", 71.2833, "C85", "C"]
]}
```

- call the endpoint

```
import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data =  {
  "Inputs": {
    "data": [
      {
        "PassengerId": 1,
        "Pclass": "3",
        "Name": "Braund Mr Owen Harris",
        "Sex": "male",
        "Age": 22,
        "SibSp": "1",
        "Parch": "0",
        "Ticket": "A/5 21171",
        "Fare": 22.5,
        "Cabin": "C76",
        "Embarked": "S"
      }
    ]
  },
  "GlobalParameters": {
    "method": "predict"
  }
}

body = str.encode(json.dumps(data))

url = 'https://xxxxxxxxxxxxxxxxxxxxxx.centralus.inference.ml.azure.com/score'
api_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' # Replace this with the API key for the web service

# The azureml-model-deployment header will force the request to go to a specific deployment.
# Remove this header to have the request observe the endpoint traffic rules
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'titanic-deploy' }

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/titsdkv2-11.jpg "Architecture")

- Predict_proba

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/titsdkv2-12.jpg "Architecture")

- get endpoint details

```
# Get the details for online endpoint
endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

# existing traffic details
print(endpoint.traffic)

# Get the scoring URI
print(endpoint.scoring_uri)
```

- Now delete the endpoint

```
ml_client.online_endpoints.begin_delete(name=online_endpoint_name)
```