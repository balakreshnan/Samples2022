# Azure ML SDK V2 sample for forecasting a time series

## Introducing Azure ML SDK V2

## Prerequisites

- Azure Account
- Azure Machine learning resource
- Default AML storage
- Create Training, Test and Validation folder
- MLTable Config for above folders
- Install preview features
- Remember print version for preview is only available using _version.VERSION

```
import azure.ai.ml
print(azure.ai.ml._version.VERSION)
```

## Code

### Code to connect existing workspace

```
# import required libraries
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential
```

- now specificy the workspace name and subscription id

```
# Enter details of your AML workspace
subscription_id = "xxxxxxxxxxxxxxxxxxxxxxxxx"
resource_group = "rgname"
workspace = "workspacename"
```

- Now call the workspace

```
# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)
```

### Now lets create a Training experiment

- Import the necessary libraries

```
# Import required libraries
from azure.ai.ml import MLClient
```

- Get Default Authentication

```
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    # This will open a browser page for
    credential = InteractiveBrowserCredential()
```

- Now load the workspace

```
try:
    ml_client = MLClient.from_config(credential=credential)
except Exception as ex:
    # NOTE: Update following workspace information if not correctly configure before
    client_config = {
        "subscription_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "resource_group": "rgname",
        "workspace_name": "workspacename",
    }

    if client_config["subscription_id"].startswith("<"):
        print(
            "please update your <SUBSCRIPTION_ID> <RESOURCE_GROUP> <AML_WORKSPACE_NAME> in notebook cell"
        )
        raise ex
    else:  # write and reload from config file
        import json, os

        config_path = "../.azureml/config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as fo:
            fo.write(json.dumps(client_config))
        ml_client = MLClient.from_config(credential=credential, path=config_path)
print(ml_client)
```

- Now create the compute target

```
from azure.ai.ml.entities import AmlCompute

# specify aml compute name.
cpu_compute_target = "cpu-cluster"

try:
    ml_client.compute.get(cpu_compute_target)
except Exception:
    print("Creating a new cpu compute target...")
    compute = AmlCompute(
        name=cpu_compute_target, size="STANDARD_D2_V2", min_instances=0, max_instances=4
    )
    ml_client.compute.begin_create_or_update(compute)
```

- lets include other libraries

```
# import required libraries
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from azure.ai.ml import MLClient, Input, command, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.automl import forecasting
from azure.ai.ml.entities._job.automl.tabular.forecasting_settings import ForecastingSettings

# add environment variable to enable private preview feature
import os
os.environ["AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED"] = "true"
```

```
try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()
```

- Get cluster name

```
# Get a handle to workspace
ml_client = MLClient.from_config(credential=credential)

# Retrieve an already attached Azure Machine Learning Compute.
cluster_name = "cpu-cluster"
print(ml_client.compute.get(cluster_name))
```

- Now setup pipeline

```
# Define pipeline
@pipeline(
    description="AutoML Forecasting Pipeline",
    )
def automl_forecasting(
    forecasting_train_data,
):
    # define forecasting settings
    forecasting_settings = ForecastingSettings(time_column_name="timeStamp", forecast_horizon=12, frequency="H", target_lags=[ 12 ], target_rolling_window_size=4)
    
    # define the automl forecasting task with automl function
    forecasting_node = forecasting(
        training_data=forecasting_train_data,
        target_column_name="demand",
        primary_metric="normalized_root_mean_squared_error",
        n_cross_validations=2,
        forecasting_settings=forecasting_settings,
        # currently need to specify outputs "mlflow_model" explictly to reference it in following nodes 
        outputs={"best_model": Output(type="mlflow_model")},
    )
    
    forecasting_node.set_limits(timeout_minutes=180)

    command_func = command(
        inputs=dict(
            automl_output=Input(type="mlflow_model")
        ),
        command="ls ${{inputs.automl_output}}",
        environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1"
    )
    show_output = command_func(automl_output=forecasting_node.outputs.best_model)


data_folder = "./data"
pipeline = automl_forecasting(
    forecasting_train_data=Input(path=f"{data_folder}/training-mltable-folder/", type="mltable"),
)

# set pipeline level compute
pipeline.settings.default_compute="cpu-cluster"
```

- Now run the pipeline

```
# submit the pipeline job
pipeline_job = ml_client.jobs.create_or_update(
    pipeline, experiment_name="pipeline_samples"
)
pipeline_job
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/amlforecast1.jpg "Architecture")

- Now get the URL to watch the pipeline job execute

```
# Wait until the job completes
ml_client.jobs.stream(pipeline_job.name)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/amlforecast2.jpg "Architecture")

- Wait for it to complete
- Monitor the job status

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/amlforecast3.jpg "Architecture")

- Final pipeline output

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/amlforecast4.jpg "Architecture")