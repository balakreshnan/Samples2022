# Azure Machine Learning Automated ML with Mlflow

## Use MLFlow with Azure Machine Learning and Automated ML

## Pre-requisites

- Azure Subscription
- Azure Machine Learning Workspace
- Azure Storage Account
- Create a data folder
- upload the data file to the data folder

## Libraries

- Install Azure.ai.ml

```
pip install azure.ai.ml
```

- Install mlflow

```
pip install azure.ai.ml
```

## Code

- Install libraries

```
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

- Authenticate to Azure machine learning workspace

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
    subscription_id = "xxxxxxxxxxxxxxx"
    resource_group = "rgname"
    workspace = "wkspacename"
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)
```

- Display workspace

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

- Create data set

```
my_training_data_input = Input(
    type=AssetTypes.MLTABLE, path="./data/"
)
my_training_data_test = Input(
    type=AssetTypes.MLTABLE, path="./data/"
)
my_training_data_validate = Input(
    type=AssetTypes.MLTABLE, path="./data/"
)
```

- Set the experiment name

```
# General job parameters
compute_name = "cpu-cluster"
max_trials = 5
exp_name = "automlv2-Titanic-experiment"
```

- Configure the experiment

```
classification_job = automl.classification(
    compute=compute_name,
    experiment_name=exp_name,
    training_data=my_training_data_input,
    target_column_name="Survived",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True,
    tags={"my_custom_tag": "Titanic Training"},
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

- Run the experiment

```
# Submit the AutoML job
returned_job = ml_client.jobs.create_or_update(
    classification_job
)  # submit the job to the backend

print(f"Created job: {returned_job}")
```

- Stream logs

```
ml_client.jobs.stream(returned_job.name)
```

- import mlflow

```
import mlflow

# Obtain the tracking URL from MLClient
MLFLOW_TRACKING_URI = ml_client.workspaces.get(
    name=ml_client.workspace_name
).mlflow_tracking_uri

print(MLFLOW_TRACKING_URI)
```

- Set the tracking URI

```
# Set the MLFLOW TRACKING URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))
```

- invoke MLflow

```
from mlflow.tracking.client import MlflowClient

# Initialize MLFlow client
mlflow_client = MlflowClient()
```

- Get the run id

```
job_name = returned_job.name

# Example if providing an specific Job name/ID
# job_name = "xxxxxx"

# Get the parent run
mlflow_parent_run = mlflow_client.get_run(job_name)

print("Parent Run: ")
print(mlflow_parent_run)
```

- Get the child runs

```
best_child_run_id = mlflow_parent_run.data.tags["automl_best_child_run_id"]
print("Found best child run id: ", best_child_run_id)

best_run = mlflow_client.get_run(best_child_run_id)

print("Best child run: ")
print(best_run)
```

- Metrics

```
best_run.data.metrics
```

- Directory

```
import os

# Create local folder
local_dir = "./artifact_downloads"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

# Download run's artifacts/outputs
local_path = mlflow_client.download_artifacts(
    best_run.info.run_id, "outputs", local_dir
)
print("Artifacts downloaded in: {}".format(local_path))
print("Artifacts: {}".format(os.listdir(local_path)))
```

- list dorectories

```
os.listdir("./artifact_downloads/outputs/mlflow-model")
```

- Deploy the endpoint

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

- Create the Endpoint

```
import datetime

online_endpoint_name = "titanic-" + datetime.datetime.now().strftime("%m%d%H%M%f")

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="this is a sample online endpoint for mlflow model",
    auth_mode="key",
    tags={"foo": "bar"},
)
ml_client.begin_create_or_update(endpoint)
```

- Register the model

```
model_name = "titanic-model"
model = Model(
    path=f"azureml://jobs/{best_run.info.run_id}/outputs/artifacts/outputs/model.pkl",
    name=model_name,
    description="my sample mlflow model",
)

# for downloaded file
# model = Model(path="artifact_downloads/outputs/model.pkl", name=model_name)

registered_model = ml_client.models.create_or_update(model)
registered_model.id
```

- Environment

```
env = Environment(
    name="automl-tabular-env",
    description="environment for automl inference",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210727.v1",
    conda_file="artifact_downloads/outputs/conda_env_v_1_0_0.yml",
)
```

- Conda configuration

```
code_configuration = CodeConfiguration(
    code="artifact_downloads/outputs/", scoring_script="scoring_file_v_2_0_0.py"
)
```

- Managed endpoint configuration

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

- deploy the endpoint

```
ml_client.online_deployments.begin_create_or_update(deployment)
```

- Test the inference

```
# test the blue deployment with some sample data
ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name="titanic-deploy",
    request_file="sample-titanic-data.json",
)
```

- Sample json file

```
{
  "Inputs": {
    "data": [
      {
        "PassengerId": 10,
        "Pclass": 1,
        "Name": "Name goes here",
        "Sex": "Male",
        "Age": 35.0,
        "SibSp": 2,
        "Parch": 1,
        "Ticket": "F123",
        "Fare": 10.0,
        "Cabin": "S",
        "Embarked": "E"
      }
    ]
  },
  "GlobalParameters": {
    "method": "predict"
  }
}
```

- Delete the endpoint

```
ml_client.online_endpoints.begin_delete(name=online_endpoint_name)
```