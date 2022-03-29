# Automated ML sdk training

## Train model using automl with sdk code approach

## Prerequisites

- Azure account
- Azure ML account
- Azure Storage
- Create a folder for Training data, Batch score input, Batch score output
- Create a compute cluster
- Need notebook or jupyter lab or jupyter to write code

## Code

- import libraries

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

- Get the workspace

```
import azureml.core
from azureml.core import Workspace, Datastore

ws = Workspace.from_config()
```

- Load training data set first

```
dataset_name = 'titanictraining3'

# Get a dataset by name
titanic_ds = Dataset.get_by_name(workspace=ws, name=dataset_name)

# Load a TabularDataset into pandas DataFrame
df = titanic_ds.to_pandas_dataframe()
```

- Split the data set for training and testing

```
import pandas as pd
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)
label = "survived"
```

- Create compute

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

- Create the automl configuration

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
                             #validation_data = validation_dataset,
                             **automl_settings
                            )
```

- Setup the automated ml experiment name

```
from azureml.core.experiment import Experiment

ws = Workspace.from_config()

# Choose a name for the experiment and specify the project folder.
experiment_name = 'Titanictraining3'
project_folder = './titanic3/automl-classification'

experiment = Experiment(ws, experiment_name)
```

- submit the experiment

```
run = experiment.submit(automl_config, show_output=True)
run.wait_for_completion()
```

- show results

```
from azureml.widgets import RunDetails
RunDetails(run).show()
```

- Now next step is to create batch scoring

https://github.com/balakreshnan/Samples2022/blob/main/AzureML/automlbatchscore.md