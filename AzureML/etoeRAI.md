# End to End Responsible AI with AML pipelines

## End to End classification with Responsible AI and Azure ML

## Prerequisites

- Azure Account
- Storage Account
- Azure ML

## Setup

- Log into Azure ML workspace
- Create a notebook with Python 3.8 with Azure ML SDK
- Let install

```
!pip install raiwidgets
!pip install --upgrade raiwidgets
!pip install --upgrade pandas
```

- Restart the kernel after installation
- Test the install

```
from raiwidgets import ResponsibleAIDashboard
from responsibleai import RAIInsights
```

## Code/Steps

- Check the AML version

```
import azureml.core

print("SDK version:", azureml.core.VERSION)
```

- Send diagnotics telemetry

```
from azureml.telemetry import set_diagnostics_collection

set_diagnostics_collection(send_diagnostics=True)
```

- Log into Azure ML workspace

```
import azureml.core
from azureml.core import Workspace
import pandas as pd

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))
```

- Load data
- Make sure have the titanic.csv in the corresponding folder

```
df= pd.read_csv('./Data/titanic.csv')
print(df.shape)
print(df.columns)
```

```
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
df.isnull().sum()
```

- Convert cabin to loc

```
df['Loc']= df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'X')
```

- Drop columns which we don't need

```
df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
```

```
df.loc[:,'GroupSize'] = 1 + df['SibSp'] + df['Parch']
```

- Fill S for empty values

```
df['Embarked'] = df['Embarked'].fillna('S')
```

- Clean up the dataset

```
LABEL = 'Survived'
columns_to_keep = ['Pclass', 'Sex','Age', 'Fare', 'Embared', 'Deck', 'GroupSize']
columns_to_drop = ['Name','SibSp', 'Parch', 'Survived']
df_train = df
df = df_train.drop(['Name','SibSp', 'Parch', 'PassengerId'], axis=1)

df.head(5)
```

- Create a training folder

```
import os
script_folder = os.path.join(os.getcwd(), "train_remote")
print(script_folder)
os.makedirs(script_folder, exist_ok=True)
```

- validate if file is uploaded

```
df.to_csv('./train_remote/titanic.csv')
df.head(2)
```

- Create a dataset

```
from azureml.core import Dataset

#use default datastore retrieved from the workspace through the AML SDK
default_ds = ws.get_default_datastore()


default_ds.upload_files(files=['./train_remote/titanic.csv'], # Upload the diabetes csv files in /data
                        target_path= 'Titanic-data', # Put it in a folder path in the datastore
                        overwrite=True, # Replace existing files of the same name
                        show_progress=True)
#Create a tabular dataset from the path on the datastore 
dataset = Dataset.Tabular.from_delimited_files(default_ds.path('Titanic-data/titanic.csv'))

# Register the dataset
try:
    tab_data_set = dataset.register(workspace=ws, 
                                name= 'Titanic-tabular-dataset',
                                description='Tintanic data',
                                tags = {'format':'csv'},
                                create_new_version=True)
    print('Dataset registered.')
except Exception as ex:
        print(ex)
```

- Now read from dataset

```
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = 'xxxxxxxxxxxxxxxxxxxx'
resource_group = 'RGName'
workspace_name = 'AMLWorkspaceName'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='Titanic-tabular-dataset')
dataset.to_pandas_dataframe()
```

- Create a Experiment

```
from azureml.core.experiment import Experiment
experiment = Experiment(ws, 'titanic_remote_compute')
```

- Create a folder to store Training Scripts

```
import os
script_folder = os.path.join(os.getcwd(), "train")
print(script_folder)
os.makedirs(script_folder, exist_ok=True)
```

- Create training script

```
%%writefile $script_folder/training.py

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

from raiwidgets import ResponsibleAIDashboard
from responsibleai import RAIInsights

# Calculate model performance metrics
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration

def getRuntimeArgs():
    parser = argparse.ArgumentParser()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str)
    args = parser.parse_args()
    return args

def buildpreprocessorpipeline(X_raw):
    categorical_features = X_raw.select_dtypes(include=['object']).columns
    numeric_features = X_raw.select_dtypes(include=['float','int64']).columns

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value="missing")),
                                              ('onehotencoder', OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore'))])
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ], remainder="drop")
    
    return preprocessor

def model_train(LABEL, df, run):  
    y_raw = df[LABEL]
    X_raw = df.drop([LABEL], axis=1)
    
     # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.3, random_state=0)
    
    lg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
    preprocessor = buildpreprocessorpipeline(X_train)
    
    #estimator instance
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', lg)])

    model = clf.fit(X_train, y_train)
    
    
    # calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('AUC: ' + str(auc))
    run.log('AUC', np.float(auc))

    
    # calculate test accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)
    run.log('Accuracy', np.float(acc))

    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    fig = plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    run.log_image(name = "ROC", plot = fig)
    plt.show()

    # plot confusion matrix
    # Generate confusion matrix
    cmatrix = confusion_matrix(y_test, y_hat)
    cmatrix_json = {
        "schema_type": "confusion_matrix",
           "schema_version": "v1",
           "data": {
               "class_labels": ["0", "1"],
               "matrix": [
                   [int(x) for x in cmatrix[0]],
                   [int(x) for x in cmatrix[1]]
               ]
           }
    }
    
    run.log_confusion_matrix('ConfusionMatrix_Test', cmatrix_json)
    
    os.makedirs('outputs', exist_ok=True)
    
    
    model_file = os.path.join('outputs', 'titanic_model.pkl')
    joblib.dump(value=model, filename=model_file)
    
    run.upload_file(name='titanic_model.pkl', path_or_stream=model_file)
    run.log('accuracy', acc)
    run.set_tags({ 'Accuracy' : np.float(acc)})
    
    # Register the model
    print('Registering model...')
    run.register_model(model_path='titanic_model.pkl', model_name= 'titanic-model',
                   tags={'Model Type':'Logistic Regresssion'},
                   properties={'AUC': np.float(auc), 'Accuracy': np.float(acc)})
    
    #features = "'Column1', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Loc', 'GroupSize'"
    #df1 = df.drop('Survived', inplace=true)
    #df1 = df
    columns = ['Survived']
    #df = df.drop('Survived', axis=1, inplace=True)
    df = df.drop(columns, axis=1)
    print(df.columns)
    client = ExplanationClient.from_run(run)
    # Explain predictions on your local machine
    tabular_explainer = TabularExplainer(model, X_train, features=df.columns)

    # Explain overall model predictions (global explanation)
    # Passing in test dataset for evaluation examples - note it must be a representative sample of the original data
    # x_train can be passed as well, but with more examples explanations it will
    # take longer although they may be more accurate
    global_explanation = tabular_explainer.explain_global(X_test)

    # Uploading model explanation data for storage or visualization in webUX
    # The explanation can then be downloaded on any compute
    comment = 'Global explanation on regression model trained on boston dataset'
    client.upload_model_explanation(global_explanation, comment=comment, model_id=model)
    categorical_features = X_raw.select_dtypes(include=['object']).columns
    target_feature = LABEL
    train_data = X_train.copy()
    test_data = X_test.copy()
    train_data[target_feature] = y_train
    test_data[target_feature] = y_test
    #data.feature_names

    rai_insights = RAIInsights(model, train_data, test_data, LABEL, 'classification', 
                               categorical_features=['Sex','Embarked','Loc'])
    # Interpretability
    rai_insights.explainer.add()
    # Error Analysis
    rai_insights.error_analysis.add()
    # Counterfactuals: accepts total number of counterfactuals to generate, the range that their label should fall under, 
    # and a list of strings of categorical feature names
    rai_insights.counterfactual.add(total_CFs=20, desired_class='opposite')
    rai_insights.compute()
    ResponsibleAIDashboard(rai_insights)


    return model, auc, acc
    # Save the trained model
    
    
def main():
    # Create an Azure ML experiment in your workspace
    args = getRuntimeArgs()
    
    run = Run.get_context()
    client = ExplanationClient.from_run(run)

    dataset_dir = './dataset/'
    os.makedirs(dataset_dir, exist_ok=True)
    ws = run.experiment.workspace
    print(ws)
    

    print("Loading Data...")
    dataset = Dataset.get_by_id(ws, id=args.input_data)
    # Load a TabularDataset & save into pandas DataFrame
    df = dataset.to_pandas_dataframe()
    
    print(df.head(5))
 
    model, auc, acc = model_train('Survived', df, run)
    
    #os.makedirs('outputs', exist_ok=True)
    
    
    #model_file = os.path.join('outputs', 'titanic_model.pkl')
    #joblib.dump(value=model, filename=model_file)
    
    #run.upload_file(name='titanic_model.pkl', path_or_stream=model_file)
    #run.log('accuracy', acc)
    #run.set_tags({ 'Accuracy' : np.float(acc)})
    
    # Register the model
    #print('Registering model...')
    #run.register_model(model_path='titanic_model.pkl', model_name= 'titanic-model',
    #               tags={'Model Type':'Logistic Regresssion'},
    #               properties={'AUC': np.float(auc), 'Accuracy': np.float(acc)})
   

    run.complete()

if __name__ == "__main__":
    main()
```

- Create a Environment file

```
%%writefile $script_folder/experiment_env.yml
name: experiment_env
dependencies:
  # The python interpreter version.
  # Currently Azure ML only supports 3.5.2 and later.
- python=3.6.2
- scikit-learn
- ipykernel
- matplotlib
- pandas
- pip
- pip:
  - azureml-defaults
  - pyarrow
  - interpret
  - azureml-interpret
  - lightgbm
  - raiwidgets
```

- Create Environment object and supply the above file

```
from azureml.core import Environment

# Create a Python environment for the experiment (from a .yml file)
experiment_env = Environment.from_conda_specification('experiment-env', script_folder + "/experiment_env.yml")

# Let Azure ML manage dependencies
experiment_env.python.user_managed_dependencies = False 

# Print the environment details
print(experiment_env.name, 'defined.')
print(experiment_env.python.conda_dependencies.serialize_to_string())
```

- Now run the experiment in local mode

```
import azureml.core.runconfig
from azureml.core import Environment, Experiment
from azureml.core import ScriptRunConfig
from azureml.widgets import RunDetails

# Get the training dataset
titanic_ds = ws.datasets.get('Titanic-tabular-dataset')

# Create a script config
script_config = ScriptRunConfig(source_directory=script_folder,
                                script='training.py',
                                arguments=['--input-data', titanic_ds.as_named_input('titanic')], # Reference to dataset
                                environment=experiment_env) 

# submit the experiment
run = experiment.submit(config=script_config)
RunDetails(run).show()
run.wait_for_completion()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/raietoe1.jpg "Architecture")

- Local run is make sure the code is working
- Actual code should be run in remote compute

- Now create remote computer cluster

```
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cluster_name = "cpu-cluster"

try:
    # Check for existing compute target
    training_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)
```

- Now run the experiment in remote mode

```
import azureml.core.runconfig
from azureml.core import Environment, Experiment
from azureml.core import ScriptRunConfig
from azureml.widgets import RunDetails

# Get the training dataset
titanic_ds = ws.datasets.get('Titanic-tabular-dataset')

# Create a script config
script_config = ScriptRunConfig(source_directory=script_folder,
                                script='training.py',
                                arguments=['--input-data', titanic_ds.as_named_input('titanic')], # Reference to dataset
                                environment=experiment_env,
                                compute_target=cluster_name)

# submit the experiment
run = experiment.submit(config=script_config)
RunDetails(run).show()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/raietoe2.jpg "Architecture")

- Now run the Hyperdrive to optimize the model

```
from azureml.core.run import Run
run_logger = Run.get_context()
#run_logger.log("accuracy", float(accuracy))
```

- Set the early termination

```
from azureml.train.hyperdrive import BanditPolicy
early_termination_policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5)
```

- load the dataset to get dataset id

```
ds = Dataset.get_by_name(ws, name="Titanic-tabular-dataset")
```

- now run the Hyperdrive

```
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.parameter_expressions import choice
    
#arguments=['--input-data', titanic_ds.as_named_input('titanic')], # Reference to dataset

titanic_ds = ws.datasets.get('Titanic-tabular-dataset')

# Create a script config
script_config = ScriptRunConfig(source_directory=script_folder,
                                script='training.py',
                                arguments=['--input-data', titanic_ds.as_named_input('titanic')], # Reference to dataset
                                environment=experiment_env,
                                compute_target=cluster_name) 

param_sampling = RandomParameterSampling( {
    "input-data": ds.id
    }
)

early_termination_policy = BanditPolicy(slack_factor=0.15, evaluation_interval=1, delay_evaluation=10)

hd_config = HyperDriveConfig(run_config=script_config,
                             hyperparameter_sampling=param_sampling,
                             policy=early_termination_policy,
                             primary_metric_name="accuracy",
                             primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                             max_total_runs=4,
                             max_concurrent_runs=4)
hyperdrive_run = experiment.submit(hd_config)
from azureml.widgets import RunDetails
RunDetails(hyperdrive_run).show()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/raietoe3.jpg "Architecture")

- Get the best model

```
best_run = hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
parameter_values = best_run.get_details()['runDefinition']['arguments']

print('Best Run Id: ', best_run.id)
print('\n Accuracy:', best_run_metrics['accuracy'])
```

- Register the model

```
model = best_run.register_model(model_name='titanic-model', model_path='outputs/titanic_model.pkl')
```

- Now deploy the model
- create a deployment directory

```
import os

# Create a folder for the experiment files
deploy_aci_folder = 'deploy-aci'
os.makedirs(deploy_aci_folder, exist_ok=True)
print(deploy_aci_folder, 'folder created')
```

- load the model to deploy

```
model_name = 'titanic-model'
print(model_name)
model = ws.models[model_name]
print(model.name, 'version', model.version)
```

- load the dataset to predict

```
import joblib
from azureml.core import Dataset
model.download(target_dir='./deploy-aci', exist_ok=True)
dataset = Dataset.get_by_name(ws, name='Titanic-tabular-dataset')
df = dataset.to_pandas_dataframe().head(1)
dftest = df.drop(['Survived'], axis=1)
loaded_model = joblib.load('deploy-aci/titanic_model.pkl')
print('Columns needed for inference:')
print(dftest.columns)
y_hat = loaded_model.predict(dftest)
print('Prediction')
print(y_hat)
```

- Create score file

```
%%writefile $deploy_aci_folder/score.py

import os 
import json
import joblib
from pandas import json_normalize
import pandas as pd

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'titanic_model.pkl')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    dict= json.loads(raw_data)
    df = json_normalize(dict['raw_data']) 
    y_pred = model.predict(df)
    print(type(y_pred))
    
    result = {"result": y_pred.tolist()}
    return result
```

- create the inference environment

```
%%writefile $deploy_aci_folder/experiment_env.yml
name: experiment_env
dependencies:
  # The python interpreter version.
  # Currently Azure ML only supports 3.5.2 and later.
- python=3.8
- scikit-learn
- ipykernel
- matplotlib
- pandas
- pip
- pip:
  - azureml-defaults
  - pyarrow
```

- create the environment

```
from azureml.core import Environment

# Create a Python environment for the experiment (from a .yml file)
experiment_env = Environment.from_conda_specification("experiment-env", deploy_aci_folder + "/experiment_env.yml")

# Let Azure ML manage dependencies
experiment_env.python.user_managed_dependencies = False
# register the environment
experiment_env.register(workspace=ws)
# Print the environment details
print(experiment_env.name, 'defined.')
print(experiment_env.python.conda_dependencies.serialize_to_string())
```

- Set ACI configuration for inferencing

```
from azureml.core.webservice import AciWebservice
aci_config = AciWebservice.deploy_configuration(
            cpu_cores = 1, 
            memory_gb = 2, 
            tags = {'model': 'titanic model'},
            auth_enabled=True,
            enable_app_insights=True,
            collect_model_data=True)
```

- Deploy the model

```
from azureml.core.model import InferenceConfig
from azureml.core import Model
#get the model object from the workspace
model = Model(ws, model_name)
#get the environment from the workspace
env = Environment.get(ws, 'experiment-env')

inference_config = InferenceConfig(source_directory=deploy_aci_folder,
                                   entry_script='score.py',
                                   environment= env)

# Deploy the model as a service
print('Deploying model...')
service_name = "titanic-service"
service = Model.deploy(ws, service_name, [model], inference_config, aci_config, overwrite=True)
```

```
service.wait_for_deployment(True)
print(service.state)
```

- get endpoint details

```
endpoint = service.scoring_uri
print(endpoint)
keys = service.get_keys()
selected_key = keys[0]
print(selected_key)
```

- get details

```
from azureml.core import Webservice
websrv = Webservice(ws, 'titanic-service')
endpoint = websrv.scoring_uri
print(endpoint)
keys = websrv.get_keys()
selected_key = keys[0]
print(selected_key)
```

- predict results

```
import json
url = endpoint
api_key = selected_key # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
import requests

def MakePrediction(df):
    endpoint_url = url
    body = df.to_json(orient='records') 
    body = '{"raw_data": ' + body + '}'
    print(body)
    r = requests.post(endpoint_url, headers=headers, data=body)
    return (r.json())


dataset = Dataset.get_by_name(ws, name='Titanic-tabular-dataset')
df = dataset.to_pandas_dataframe().head(5)
dftest = df.drop(['Survived'], axis=1)

results = MakePrediction(dftest)

val = results['result']
print('')
print('predictions')
print(val)
```

```
print(service.state)
```

- Now delete the file

```
service.delete()
```