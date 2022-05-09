# Azure Machine learning running spacy NLP

## Build an end to end pipeline using Azure machine learning for training and scoring

## Prerequisites

- Azure Account
- Azure Storage
- Azure Machine Learning Service


## Code

- This sample code is to show how to create and run training and inferencing aml pipeline using sdk
- Not an actual implementation
- Training and inferencing code are samples and not ready for production use
- Context is ready for production use
- Tested this code in python 3.8 with azure ML kernel
- Lets configure the workspace to run
- The below code assumes input data is set ADLS gen2 with dataset created for input and output.
- When we have input and output in storage, any consuming application can take the results
- The below code is for batch processing only.

```
import azureml.core
from azureml.core import Workspace, Datastore

ws = Workspace.from_config()
```

- Next is configure the workspace default datastore

```
# Default datastore 
def_data_store = ws.get_default_datastore()

# Get the blob storage associated with the workspace
def_blob_store = Datastore(ws, "workspaceblobstore")

# Get file storage associated with the workspace
def_file_store = Datastore(ws, "workspacefilestore")
```

- Next create compute cluster

```
from azureml.core.compute import ComputeTarget, AmlCompute

compute_name = "cpu-cluster"
vm_size = "STANDARD_NC6"
if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('Found compute target: ' + compute_name)
else:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,  # STANDARD_NC6 is GPU-enabled
                                                                min_nodes=0,
                                                                max_nodes=4)
    # create the compute target
    compute_target = ComputeTarget.create(
        ws, compute_name, provisioning_config)

    # Can poll for a minimum number of nodes and for a specific timeout.
    # If no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=20)

    # For a more detailed view of current cluster status, use the 'status' property
    print(compute_target.status.serialize())
```

- We are only using CPU cluster. Option for GPU is available if needed
- Import AML libaries

```
from azureml.core import Dataset
from azureml.data.dataset_factory import DataType
from azureml.pipeline.steps import PythonScriptStep
from azureml.data import OutputFileDatasetConfig
from azureml.core import Workspace, Datastore

datastore = ws.get_default_datastore()
```

- Create a Environment configuration for compute

```
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Environment 

aml_run_config = RunConfiguration()
# `compute_target` as defined in "Azure Machine Learning compute" section above
aml_run_config.target = compute_target

USE_CURATED_ENV = True
if USE_CURATED_ENV :
    curated_environment = Environment.get(workspace=ws, name="AzureML-Tutorial")
    aml_run_config.environment = curated_environment
else:
    aml_run_config.environment.python.user_managed_dependencies = False
    
    # Add some packages relied on by data prep step
    aml_run_config.environment.python.conda_dependencies = CondaDependencies.create(
        conda_packages=['pandas','scikit-learn','seaborn','tqdm'], 
        pip_packages=['azureml-sdk', 'azureml-dataprep[fuse,pandas]','seaborn','tqdm', 'spacy'], 
        pin_sdk_version=False)
```

- Now let's write the train.py code
- Create a new file as Text file and rename as train.py

```

import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'spacy'])
subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])

import spacy
nlp = spacy.load('en_core_web_sm')

about_text = ('Gus Proto is a Python developer currently'
              ' working for a London-based Fintech'
              ' company. He is interested in learning'
              ' Natural Language Processing.')
about_doc = nlp(about_text)
sentences = list(about_doc.sents)
len(sentences)

for sentence in sentences:
    print (sentence)
```

- Above code should be train folder
- If there is none please create a new one
- The above code doesn't do much than just print the sentences
- Above code can be edited to fit your scenario
- Also code to add run information can also be added
- Next we setup the training pipeline

```
train_source_dir = "./train"
train_entry_point = "train.py"


    
train_step = PythonScriptStep(
    script_name=train_entry_point,
    source_directory=train_source_dir,
    ##arguments=["--input_data", ds_input],
    compute_target=compute_target, # , "--training_results", training_results
    runconfig=aml_run_config,
    allow_reuse=False
)
```

- Create the pipeline steps

```
# list of steps to run (`compare_step` definition not shown)
compare_models = [train_step]

from azureml.pipeline.core import Pipeline

# Build the pipeline
pipeline1 = Pipeline(workspace=ws, steps=train_step)
```

- Validate the pipeline

```
pipeline1.validate()
print("Pipeline validation complete")
```

- Run the training pipeline

```
from azureml.core import Experiment

# Submit the pipeline to be run
pipeline_run1 = Experiment(ws, 'Spacy_Pipeline_Notebook').submit(pipeline1)
pipeline_run1.wait_for_completion()
```

- Display output

```
from azureml.widgets import RunDetails

RunDetails(pipeline_run1).show()
```

- Next create pipeline for retraining using synapse integrate or Azure data factory

```
from azureml.pipeline.core.graph import PipelineParameter

pipeline_param = PipelineParameter(
  name="pipeline_arg",
  default_value=10)
```

- publish the pipeline

```
published_pipeline1 = pipeline_run1.publish_pipeline(
     name="Published_Spacy_Pipeline_Notebook",
     description="Spacy_Pipeline_Notebook Published Pipeline Description",
     version="1.0")
```

- Pipeline parameters is not used in the above training script just to show how to pass
- To execute the above pipeline first we need permission
- Setup Azure Service principal
- Provide contributor access to the AML workspace to run the pipeline

```
from azureml.core.authentication import TokenAuthentication, Audience

# This is a sample method to retrieve token and will be passed to TokenAuthentication
def get_token_for_audience(audience):
    from adal import AuthenticationContext
    client_id = "clientid"
    client_secret = "xxxxxxxxxxxxxxxx"
    tenant_id = "tenantid"
    auth_context = AuthenticationContext("https://login.microsoftonline.com/{}".format(tenant_id))
    resp = auth_context.acquire_token_with_client_credentials(audience,client_id,client_secret)
    token = resp["accessToken"]
    return token


token_auth = TokenAuthentication(get_token_for_audience=get_token_for_audience)
```

- Create the authoriaztion header

```
headerInfo = {'Authorization': 'Bearer ' + aad_token + ''}
```

- Now invoke the published pipeline

```
from azureml.pipeline.core import PublishedPipeline
import requests

response = requests.post(published_pipeline1.endpoint, 
                         headers=headerInfo,
                         json={"ExperimentName": "Published_Spacy_Pipeline_Notebook",
                               "ParameterAssignments": {"pipeline_arg": 20}})
```

- Now go to experiment page and wait for experiment to complete
- Once the experiment is completed next we can try score pipeline

- First create a score.py file

```

import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'spacy'])
subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])

import spacy
nlp = spacy.load('en_core_web_sm')

about_text = ('Gus Proto is a Python developer currently'
              ' working for a London-based Fintech'
              ' company. He is interested in learning'
              ' Natural Language Processing.')
about_doc = nlp(about_text)
sentences = list(about_doc.sents)
len(sentences)

for sentence in sentences:
    print (sentence)
```

- Create a inference pipeline

```
train_source_dir = "./inference"
train_entry_point = "score.py"


    
train_step = PythonScriptStep(
    script_name=train_entry_point,
    source_directory=train_source_dir,
    ##arguments=["--input_data", ds_input],
    compute_target=compute_target, # , "--training_results", training_results
    runconfig=aml_run_config,
    allow_reuse=False
)
```

- Create the pipeline steps

```
# list of steps to run (`compare_step` definition not shown)
compare_models = [train_step]

from azureml.pipeline.core import Pipeline

# Build the pipeline
pipeline1 = Pipeline(workspace=ws, steps=train_step)
```

- Validate the pipeline

```
pipeline1.validate()
print("Pipeline validation complete")
```

- Submit the pipeline run.
- Go to experiment and wait for pipeline to complete

```
from azureml.core import Experiment

# Submit the pipeline to be run
pipeline_run1 = Experiment(ws, 'Spacy_Pipeline_Inferencing').submit(pipeline1)
pipeline_run1.wait_for_completion()
```

- Once experiment completes success, then we are good.