# Azure Machine learning Data Labeling and Object detection model (AutoML) Model

## End to end example of object detection using open source dataset using data labelling

## prerequisites

- Azure Account
- Azure storage
- Azure machine learning account
- Coco dataset download - https://cocodataset.org/#download
- Download 2017 image set and unzip
- Upload to Azure storage
- Infrastructure setup is not discussed in this article

## Process

- Define Use case
- Collect images
- Label Images
- Modelling

- Above are high level process that is involved to build or implement a end to end vision based object detection models

## Steps

- Log into Azure Machine learning workspace
- First create a data set for data labelling tool
- Create a new datastore and dataset as file dataset and point to the directory where images are stored

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe2.jpg "Architecture")

- Images files are stored in the default storage under cocotrain2017 folder

- Next Choose the default store and folder

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe3.jpg "Architecture")

- Select train2017 folder which has images
- Skip data validation

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe4.jpg "Architecture")

- Click Next and then click create

### Data Labeling

- Lets now collect the label's to use for training based on use case
- Go to AML workspace and go to Data labeling tool
- Click create Add Project

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe5.jpg "Architecture")

- now next screen

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe6.jpg "Architecture")

- disable vendor, in our case we are going to use internal people to label
- If you need assistance in labeling then please choose a partner from the list
- then click next
- Select cocotrain2017 as dataset to use for creating bounding box

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe7.jpg "Architecture")

- Then click Next
- Enable incremental refresh, this features allows us to add more images to the dataset as we collect more

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe8.jpg "Architecture")

- In our case for this totorial we are creating 3 labels
    - Person
    - food
    - animal
- then click Next

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe9.jpg "Architecture")

- Next provide instruction on how to label
- Provides URL is there is one for labelers to use that to draw bounding box
- Provide a escalation path to reach out if any questions

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe10.jpg "Architecture")

- Click Next

- Next enable ML assisted labeling
- This can help amplify and increase productivity to your labeling project
- First human labelers will label the images
- Once first set is labeled, Machine learning model learns from the labels
- Then ML model labels the images
- Human labelers can review the image and take a decision if it's valid or not
- Then submit the label for further process

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe11.jpg "Architecture")

- Then Click Create Project
- Once the project is build go to project
- When you open the project you should see the main project dashboard
- Give the link to AML data label project to your Labelers
- There are permission allowed to make sure labelers can only see project they are assigned to
- When labelers log in they should be able to click project and click Data Label

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe12.jpg "Architecture")

- Then Click Start Labeling
- in case if you want to read instructions for labeling click the instruction any time
- Each labeler can now label the images
- Once they reach like initial set usually around 75 or 100 per labeler
- ML model kicks in behind the scene and will generate labels for the images

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe13.jpg "Architecture")

- After the initial set is labeled, ML model learns from the labels
- Images that are ML labeled are then sent to human labelers for review
- Images are marked as Task prelabeled, this denotes that ML Model has already labeled the images

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe14.jpg "Architecture")

- If you are satisfied by the bounding box created then click submit
- Else adjust the bounding box and then submit
- Also validate it's as assigned the right label
- Keep doing this excercise until the entire image set is labeled
- Now lets look at the dashboard to view performance of labelers and status of labeling project

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe15.jpg "Architecture")

- In the above dashboard you can see the total images, how much labels are assigned, how many are left to be labeled
- There is also labeler performance which shows how many images each labeler has labeled
- In the bottom section you can see the ML model train run both training and inferencing
- Click on details to see the run details

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe16.jpg "Architecture")

- To review the data click on data and then review

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe17.jpg "Architecture")

- Once you are done with the project you can click on the Export
- Click Submit to create Azure ML dataset

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe18.jpg "Architecture")

### AutoVisionML code to build model

- Create a compute instance
- Create a new jupter notebook
- Go to Datasets and you should see the new exported dataset
- select the dataset usually starts with project name

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe19.jpg "Architecture")

- Go to consume section and copy the dataset information
- the above code loads the dataset for experiment
- Now create a new jupyter notebook or notebook
- import the includes

```
import os
import shutil

from azureml.core.workspace import Workspace
from azureml.core import Experiment
from azureml.core import Environment

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
```

- Load the dataset for training
- this is the dataset from data labeling project

```
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = 'xxxxx-xxxxxx-xxxxxxxxxxxxxxxx'
resource_group = 'xxxxxx'
workspace_name = 'xxxxxxxx'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='cocotrain2017_20220424_180733')
dataset.to_pandas_dataframe()
```

- Make sure name of the dataset matches the one we exported in previous section above
- Next load the workspace

```
from azureml.core.workspace import Workspace

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')
```

- Load AutoMlImage libraries

```
from azureml.train.automl import AutoMLImageConfig
from azureml.automl.core.shared.constants import ImageTask
```

- Next would be create a GPU compute

```
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# choose a name for your cluster
cluster_name = "gpu-cluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6s_V3', 
                                                           max_nodes=4)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)

# use get_status() to get a detailed status for the current cluster. 
print(compute_target.get_status().serialize())
```

- next we are going to run one model using yolov5

```
from azureml.train.automl import AutoMLImageConfig
from azureml.train.hyperdrive import GridParameterSampling, choice
from azureml.automl.core.shared.constants import ImageTask

arguments = ["--early_stopping", 1, "--evaluation_frequency", 2]

automl_image_config_yolov5 = AutoMLImageConfig(task=ImageTask.IMAGE_OBJECT_DETECTION,
                                               compute_target=compute_target,
                                               training_data=dataset,
                                               validation_data=dataset,
                                               hyperparameter_sampling=GridParameterSampling({'model_name': choice('yolov5')}),
                                               #primary_metric='mean_average_precision',
                                               iterations=1)
```

- now create a experiment and run

```
ws = Workspace.from_config()
experiment = Experiment(ws, "coco128-automl-image-object-detection")
automl_image_run = experiment.submit(automl_image_config_yolov5)
automl_image_run.wait_for_completion(wait_post_processing=True)
```

- Wait for model to run
- Get the best model

```
best_child_run = automl_image_run.get_best_child()
model_name = best_child_run.properties['model_name']
model = best_child_run.register_model(model_name = model_name, model_path='outputs/model.pt')
```

- display the model to validate

```
model
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe20.jpg "Architecture")

- now lets run few more models with different hyperparameters

```
from azureml.automl.core.shared.constants import ImageTask
from azureml.train.automl import AutoMLImageConfig
from azureml.train.hyperdrive import BanditPolicy, RandomParameterSampling
from azureml.train.hyperdrive import choice, uniform

parameter_space = {
    "model": choice(
        {
            "model_name": choice("yolov5"),
            "learning_rate": uniform(0.0001, 0.01),
            "model_size": choice("small", "medium"),  # model-specific
            #'img_size': choice(640, 704, 768), # model-specific; might need GPU with large memory
        },
        {
            "model_name": choice("fasterrcnn_resnet50_fpn"),
            "learning_rate": uniform(0.0001, 0.001),
            "optimizer": choice("sgd", "adam", "adamw"),
            "min_size": choice(600, 800),  # model-specific
            #'warmup_cosine_lr_warmup_epochs': choice(0, 3),
        },
    ),
}

tuning_settings = {
    "iterations": 2,
    "max_concurrent_iterations": 2,
    "hyperparameter_sampling": RandomParameterSampling(parameter_space),
    "early_termination_policy": BanditPolicy(
        evaluation_interval=2, slack_factor=0.2, delay_evaluation=6
    ),
}

automl_image_config = AutoMLImageConfig(
    task=ImageTask.IMAGE_OBJECT_DETECTION,
    compute_target=compute_target,
    training_data=dataset,
    validation_data=dataset,
    **tuning_settings,
)
```

- Run the model

```
automl_image_run = experiment.submit(automl_image_config)
automl_image_run.wait_for_completion(wait_post_processing=True)
```

- Shows the model run

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe21.jpg "Architecture")

- Lets look at the metrics

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe22.jpg "Architecture")

- Let's look at both runs charts for metrics
- select the 2 runs

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/objdetetoe23.jpg "Architecture")

- Explore and see other menu items available for logs, model outputs, and other
- Now you can see how easy and seamless to create label dataset and then run models
- This article doesn't include inferencing, but you can use the model to make predictions


### Inferencing

- Create a AKS GPU enabled cluster

```
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.exceptions import ComputeTargetException

# Choose a name for your cluster
aks_name = "cluster-aks-gpu"

# Check to see if the cluster already exists
try:
    aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    # Provision AKS cluster with GPU machine
    prov_config = AksCompute.provisioning_configuration(vm_size="STANDARD_NC6", 
                                                        location="eastus2")
    # Create the cluster
    aks_target = ComputeTarget.create(workspace=ws, 
                                      name=aks_name, 
                                      provisioning_configuration=prov_config)
    aks_target.wait_for_completion(show_output=True)
```

- get the model files

```
from azureml.core.model import InferenceConfig

best_child_run.download_file('outputs/scoring_file_v_1_0_0.py', output_file_path='score.py')
environment = best_child_run.get_environment()
inference_config = InferenceConfig(entry_script='score.py', environment=environment)
```

- Deploy the model

```
# Deploy the model from the best run as an AKS web service
from azureml.core.webservice import AksWebservice
from azureml.core.webservice import Webservice
from azureml.core.model import Model
from azureml.core.environment import Environment

aks_config = AksWebservice.deploy_configuration(autoscale_enabled=True,                                                    
                                                cpu_cores=1,
                                                memory_gb=50,
                                                enable_app_insights=True)

aks_service = Model.deploy(ws,
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=aks_config,
                           deployment_target=aks_target,
                           name='automl-image-test',
                           overwrite=True)
aks_service.wait_for_deployment(show_output=True)
print(aks_service.state)
```

- Code sample from - https://docs.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-image-models

## Done