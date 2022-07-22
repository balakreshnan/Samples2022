# Azure Machine learning SDK V2 - AutoML Vision

## Azure Machine learning SDK V2 AutoML Vision Experiment for Object detection with mlflow

## Prerequisites

- Azure Account
- Azure Machine learning resource
- Default AML storage
- Create Training, Test and Validation folder
- MLTable Config for above folders
- Install preview features
- Remember print version for preview is only available using _version.VERSION
- Below shows the folder structure to organize your data
- Data is available here - https://github.com/Azure/azureml-examples/tree/sdk-preview/sdk/jobs/automl-standalone-jobs/automl-image-object-detection-task-fridge-items

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/datafolderv2-1.jpg "Architecture")

- For MlTable based on the folder here is a sample training code

- Entire Code flow

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/autovision4.jpg "Architecture")

```
paths:
  - file: ./train_annotations.jsonl
transformations:
  - read_json_lines:
        encoding: utf8
        invalid_lines: error
        include_path_column: false
  - convert_column_types:
      - columns: image_url
        column_type: stream_info
```

- For Text data it could be .csv files
- File: is where the actual training file name is provided
- for here we are creating a data folder and then all the data folders are placed
- Make sure AML SDK v2 is installed

```
import azure.ai.ml
print(azure.ai.ml._version.VERSION)
```

- Install mlflow if not installed

```
pip install azureml-mlflow

pip install mlflow
```

## Code

### Code to connect existing workspace

```
# Import required libraries
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input
from azure.ai.ml.automl import ImageObjectDetectionSearchSpace
from azure.ai.ml.sweep import (
    Choice,
    Uniform,
    BanditPolicy,
)

from azure.ai.ml import automl
```

- Now lets load the existing workspace

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
    subscription_id = "subid"
    resource_group = "rgname"
    workspace = "workspacename"
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)
```

- Lets download the data for images

```
import os
import urllib
from zipfile import ZipFile

# download data
download_url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip"
data_file = "./data/odFridgeObjects.zip"
urllib.request.urlretrieve(download_url, filename=data_file)

# extract files
with ZipFile(data_file, "r") as zip:
    print("extracting files...")
    zip.extractall(path="./data")
    print("done")
# delete zip file
os.remove(data_file)
```

- Display a sample image

```
from IPython.display import Image

sample_image = "./data/odFridgeObjects/images/31.jpg"
Image(filename=sample_image)
```

- upload the images for appropriate folders

```
# Uploading image files by creating a 'data asset URI FOLDER':

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_data = Data(
    path="./data/odFridgeObjects",
    type=AssetTypes.URI_FOLDER,
    description="Fridge-items images Object detection",
    name="fridge-items-images-object-detection",
)

uri_folder_data_asset = ml_client.data.create_or_update(my_data)

print(uri_folder_data_asset)
print("")
print("Path to folder in Blob Storage:")
print(uri_folder_data_asset.path)
```

- now lets create the jsonl file for training

```
import json
import os
import xml.etree.ElementTree as ET

src_images = "./data/odFridgeObjects/"

# We'll copy each JSONL file within its related MLTable folder
training_mltable_path = "./data/training-mltable-folder/"
validation_mltable_path = "./data/validation-mltable-folder/"

train_validation_ratio = 5

# Path to the training and validation files
train_annotations_file = os.path.join(training_mltable_path, "train_annotations.jsonl")
validation_annotations_file = os.path.join(
    validation_mltable_path, "validation_annotations.jsonl"
)

# Baseline of json line dictionary
json_line_sample = {
    "image_url": uri_folder_data_asset.path,
    "image_details": {"format": None, "width": None, "height": None},
    "label": [],
}

# Path to the annotations
annotations_folder = os.path.join(src_images, "annotations")

# Read each annotation and convert it to jsonl line
with open(train_annotations_file, "w") as train_f:
    with open(validation_annotations_file, "w") as validation_f:
        for i, filename in enumerate(os.listdir(annotations_folder)):
            if filename.endswith(".xml"):
                print("Parsing " + os.path.join(src_images, filename))

                root = ET.parse(os.path.join(annotations_folder, filename)).getroot()

                width = int(root.find("size/width").text)
                height = int(root.find("size/height").text)

                labels = []
                for object in root.findall("object"):
                    name = object.find("name").text
                    xmin = object.find("bndbox/xmin").text
                    ymin = object.find("bndbox/ymin").text
                    xmax = object.find("bndbox/xmax").text
                    ymax = object.find("bndbox/ymax").text
                    isCrowd = int(object.find("difficult").text)
                    labels.append(
                        {
                            "label": name,
                            "topX": float(xmin) / width,
                            "topY": float(ymin) / height,
                            "bottomX": float(xmax) / width,
                            "bottomY": float(ymax) / height,
                            "isCrowd": isCrowd,
                        }
                    )
                # build the jsonl file
                image_filename = root.find("filename").text
                _, file_extension = os.path.splitext(image_filename)
                json_line = dict(json_line_sample)
                json_line["image_url"] = (
                    json_line["image_url"] + "images/" + image_filename
                )
                json_line["image_details"]["format"] = file_extension[1:]
                json_line["image_details"]["width"] = width
                json_line["image_details"]["height"] = height
                json_line["label"] = labels

                if i % train_validation_ratio == 0:
                    # validation annotation
                    validation_f.write(json.dumps(json_line) + "\n")
                else:
                    # train annotation
                    train_f.write(json.dumps(json_line) + "\n")
            else:
                print("Skipping unknown file: {}".format(filename))
```

- Configure train and validation folders

```
# Training MLTable defined locally, with local data to be uploaded
my_training_data_input = Input(type=AssetTypes.MLTABLE, path=training_mltable_path)

# Validation MLTable defined locally, with local data to be uploaded
my_validation_data_input = Input(type=AssetTypes.MLTABLE, path=validation_mltable_path)

# WITH REMOTE PATH: If available already in the cloud/workspace-blob-store
# my_training_data_input = Input(type=AssetTypes.MLTABLE, path="azureml://datastores/workspaceblobstore/paths/vision-classification/train")
# my_validation_data_input = Input(type=AssetTypes.MLTABLE, path="azureml://datastores/workspaceblobstore/paths/vision-classification/valid")
```

- Now set the experiment name

```
# general job parameters
compute_name = "gpu-cluster"
exp_name = "automlv2-image-object-detection-experiment"
```

- Create automl configuration

```
# Create the AutoML job with the related factory-function.

image_object_detection_job = automl.image_object_detection(
    compute=compute_name,
    experiment_name=exp_name,
    training_data=my_training_data_input,
    validation_data=my_validation_data_input,
    target_column_name="label",
    primary_metric="mean_average_precision",
    tags={"my_custom_tag": "My custom value"},
)
```

- Set the limits

```
# Set limits
image_object_detection_job.set_limits(timeout_minutes=60)
```

- now the parameters

```
# Pass the fixed settings or parameters
image_object_detection_job.set_image_model(early_stopping=True, evaluation_frequency=1)
```

- Configure parameter sweep settings

```
# Configure sweep settings
image_object_detection_job.set_sweep(
    max_trials=10,
    max_concurrent_trials=2,
    sampling_algorithm="random",
    early_termination=BanditPolicy(
        evaluation_interval=2, slack_factor=0.2, delay_evaluation=6
    ),
)
```

- now define search space

```
# Define search space
image_object_detection_job.extend_search_space(
    [
        ImageObjectDetectionSearchSpace(
            model_name=Choice(["yolov5"]),
            learning_rate=Uniform(0.0001, 0.01),
            model_size=Choice(["small", "medium"]),  # model-specific
            # image_size=Choice(640, 704, 768),  # model-specific; might need GPU with large memory
        ),
        ImageObjectDetectionSearchSpace(
            model_name=Choice(["fasterrcnn_resnet50_fpn"]),
            learning_rate=Uniform(0.0001, 0.001),
            optimizer=Choice(["sgd", "adam", "adamw"]),
            min_size=Choice([600, 800]),  # model-specific
            # warmup_cosine_lr_warmup_epochs=Choice([0, 3]),
        ),
    ]
)
```

- now time to create the experiment and submit

```
# Submit the AutoML job
returned_job = ml_client.jobs.create_or_update(
    image_object_detection_job
)  # submit the job to the backend

print(f"Created job: {returned_job}")
```

- run the experiment

```
ml_client.jobs.stream(returned_job.name)
```

- Wait for the job to complete

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/autovision1.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/autovision2.jpg "Architecture")

- Now lets track the job using mlflow

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

- enable mlflow client

```
from mlflow.tracking.client import MlflowClient

# Initialize MLFlow client
mlflow_client = MlflowClient()
```

- get jobs details

```
job_name = returned_job.name

# Example if providing an specific Job name/ID
# job_name = "salmon_camel_5sdf05xvb3"

# Get the parent run
mlflow_parent_run = mlflow_client.get_run(job_name)

print("Parent Run: ")
print(mlflow_parent_run)
```

- Print job tags

```
# Print parent run tags. 'automl_best_child_run_id' tag should be there.
print(mlflow_parent_run.data.tags)
```

- now get the best child run

```
# Get the best model's child run

best_child_run_id = mlflow_parent_run.data.tags["automl_best_child_run_id"]
print("Found best child run id: ", best_child_run_id)

best_run = mlflow_client.get_run(best_child_run_id)

print("Best child run: ")
print(best_run)
```

- Print metrics

```
import pandas as pd

pd.DataFrame(best_run.data.metrics, index=[0]).T
```

- now lets download the artifacts, models file to register

```
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

- now set model folders

```
import os

# Show the contents of the MLFlow model folder
os.listdir("./artifact_downloads/outputs/mlflow-model")
```

- Required libraries

```
# import required libraries
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
```

- Now let's create a managed online endpoint

```
# Creating a unique endpoint name with current datetime to avoid conflicts
import datetime

online_endpoint_name = "od-fridge-items-" + datetime.datetime.now().strftime(
    "%m%d%H%M%f"
)

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="this is a sample online endpoint for deploying model",
    auth_mode="key",
    tags={"foo": "bar"},
)
```

```
ml_client.begin_create_or_update(endpoint)
```

- Register the model

```
model_name = "od-fridge-items-model"
model = Model(
    path=f"azureml://jobs/{best_run.info.run_id}/outputs/artifacts/outputs/model.pt",
    name=model_name,
    description="my sample object detection model",
)

# for downloaded file
# model = Model(path="artifact_downloads/outputs/model.pt", name=model_name)

registered_model = ml_client.models.create_or_update(model)
```

```
registered_model.id
```

- Set the environment

```
env = Environment(
    name="automl-images-env",
    description="environment for automl images inference",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04",
    conda_file="artifact_downloads/outputs/conda_env_v_1_0_0.yml",
)
```

- Scoring file

```
code_configuration = CodeConfiguration(
    code="artifact_downloads/outputs/", scoring_script="scoring_file_v_1_0_0.py"
)
```

- deployment configuration

```
deployment = ManagedOnlineDeployment(
    name="od-fridge-items-deploy",
    endpoint_name=online_endpoint_name,
    model=registered_model.id,
    environment=env,
    code_configuration=code_configuration,
    instance_type="Standard_DS3_V2",
    instance_count=1,
)
```

- deploy managed endpoint

```
ml_client.online_deployments.begin_create_or_update(deployment)
```

- wait for endpoint to create

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/autovision3.jpg "Architecture")

- Now update traffic to 100%

```
# od fridge items deployment to take 100% traffic
endpoint.traffic = {"od-fridge-items-deploy": 100}
ml_client.begin_create_or_update(endpoint)
```

- now get the new REST APi to score

```
# Get the details for online endpoint
endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

# existing traffic details
print(endpoint.traffic)

# Get the scoring URI
print(endpoint.scoring_uri)
```

- Now load a validate image and score with above REST API

```
import requests

# URL for the endpoint
scoring_uri = endpoint.scoring_uri

# If the endpoint is authenticated, set the key or token
key = ml_client.online_endpoints.list_keys(name=online_endpoint_name).primary_key

sample_image = "./data/odFridgeObjects/images/1.jpg"

# Load image data
data = open(sample_image, "rb").read()

# Set the content type
headers = {"Content-Type": "application/octet-stream"}

# If authentication is enabled, set the authorization header
headers["Authorization"] = f"Bearer {key}"

# Make the request and display the response
resp = requests.post(scoring_uri, data, headers=headers)
print(resp.text)
```

- Finally delete the endpoint

```
ml_client.online_endpoints.begin_delete(name=online_endpoint_name)
```

- Wait for endpoint to delete