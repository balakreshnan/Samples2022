# Azure Machine learning Automated ML with Forecasting

## Implement Forecasting using Automated Machine learning

## Prerequisites

- Azure Account
- Azure Storage
- Azure Machine learning services

## Data Set requirements

- Time series data with Date
- Either Daily, hourly or Weekly
- Remember data has to match the above frequency
- For example for daily every day one unique record per product or other entitiy to use
- no duplicates are allowed
- split the data for training and test
- mostly use year for example i have 5 years so i take 2017-2021 as training
- 2022 for testing
- cannot have date appearing in training and testing, has to be unique dates in training and unique in testing
- upload the csv file to the sample folder as this notebook

## Notebook Code

- imports

```
import json
import logging
from datetime import datetime

import azureml.core
import numpy as np
import pandas as pd
from azureml.automl.core.featurization import FeaturizationConfig
from azureml.core import Dataset, Experiment, Workspace
from azureml.train.automl import AutoMLConfig
```

- Print the sdk version to make sure it latest
- at this time of article it was 1.42.0

```
print("You are currently using version", azureml.core.VERSION, "of the Azure ML SDK")
```

- get the workspace details

```
ws = Workspace.from_config()

# choose a name for the run history container in the workspace
experiment_name = "automl-bikeshareforecasting"

experiment = Experiment(ws, experiment_name)

output = {}
output["Subscription ID"] = ws.subscription_id
output["Workspace"] = ws.name
output["SKU"] = ws.sku
output["Resource Group"] = ws.resource_group
output["Location"] = ws.location
output["Run History Name"] = experiment_name
output["SDK Version"] = azureml.core.VERSION
pd.set_option("display.max_colwidth", None)
outputDf = pd.DataFrame(data=output, index=[""])
outputDf.T
```

- Create a compute cluster for automated ML

```
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your cluster.
amlcompute_cluster_name = "cpu-cluster"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    print("Found existing cluster, use it.")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_DS12_V2", max_nodes=4
    )
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)
```

- Now upload the csv to data store

```
datastore = ws.get_default_datastore()
datastore.upload_files(
    files=["./DM_Databrics2.csv"], target_path="dataset/", overwrite=True, show_progress=True
)
```

- Setup variables

```
target_column_name = "Qty"
time_column_name = "Date"
time_series_id_column_names = ['Plant', 'ST', 'GPN']
```

- Training data set

```
dataset = Dataset.Tabular.from_delimited_files(
    path=[(datastore, "dataset/DM_Databrics2.csv")]
).with_timestamp_columns(fine_grain_timestamp=time_column_name)

# Drop the columns 'casual' and 'registered' as these columns are a breakdown of the total and therefore a leak.
dataset = dataset.drop_columns(columns=["Unit_Price", "Revenue"])

dataset.take(5).to_pandas_dataframe().reset_index(drop=True)
```

```
# select data that occurs before a specified date
train = dataset.time_before(datetime(2021, 8, 31), include_boundary=True)
#train.to_pandas_dataframe().tail(5).reset_index(drop=True)
train.to_pandas_dataframe().drop_duplicates(['Date','Plant','ST', 'GPN'], keep='last')
```

- Now Test data set

```
test = dataset.time_after(datetime(2021, 9, 1), include_boundary=True)
#test.to_pandas_dataframe().head(5).reset_index(drop=True)
test.to_pandas_dataframe().drop_duplicates(['Date','Plant','ST', 'GPN'], keep='last')
```

- Set forecast horizon

```
forecast_horizon = 14
```

- Now configure featurization

```
featurization_config = FeaturizationConfig()
# Force the target column, to be integer type.
featurization_config.add_prediction_transform_type("Integer")
```

- Setup the automated machine learning config

```
from azureml.automl.core.forecasting_parameters import ForecastingParameters

forecasting_parameters = ForecastingParameters(
    time_column_name=time_column_name,
    forecast_horizon=forecast_horizon,
    time_series_id_column_names=time_series_id_column_names,
    country_or_region_for_holidays="US",  # set country_or_region will trigger holiday featurizer
    target_lags="auto",  # use heuristic based lag setting
    freq="D",  # Set the forecast frequency to be daily
)

automl_config = AutoMLConfig(
    task="forecasting",
    primary_metric="normalized_root_mean_squared_error",
    featurization=featurization_config,
    blocked_models=["ExtremeRandomTrees"],
    experiment_timeout_hours=0.3,
    training_data=train,
    label_column_name=target_column_name,
    compute_target=compute_target,
    enable_early_stopping=True,
    n_cross_validations=3,
    max_concurrent_iterations=4,
    max_cores_per_iteration=-1,
    verbosity=logging.INFO,
    short_series_handling_configuration="auto",
    forecasting_parameters=forecasting_parameters,
)
```

- Now run the job

```
remote_run = experiment.submit(automl_config, show_output=False)
remote_run.wait_for_completion()
```

- Get the best run

```
best_run = remote_run.get_best_child()
best_run
```

- download features names

```
# Download the JSON file locally
best_run.download_file(
    "outputs/engineered_feature_names.json", "engineered_feature_names.json"
)
with open("engineered_feature_names.json", "r") as f:
    records = json.load(f)

records
```

```
# Download the featurization summary JSON file locally
best_run.download_file(
    "outputs/featurization_summary.json", "featurization_summary.json"
)

# Render the JSON as a pandas DataFrame
with open("featurization_summary.json", "r") as f:
    records = json.load(f)
fs = pd.DataFrame.from_records(records)

# View a summary of the featurization
fs[
    [
        "RawFeatureName",
        "TypeDetected",
        "Dropped",
        "EngineeredFeatureCount",
        "Transformations",
    ]
]
```

- now test experiment

```
test_experiment = Experiment(ws, experiment_name + "_test")
```

- setup the directory

```
import os
import shutil

script_folder = os.path.join(os.getcwd(), "forecast")
os.makedirs(script_folder, exist_ok=True)
shutil.copy("forecasting_script.py", script_folder)
```

- Forecast function

```
from azureml.core import ScriptRunConfig


def run_rolling_forecast(
    test_experiment,
    compute_target,
    train_run,
    test_dataset,
    target_column_name,
    inference_folder="./forecast",
):
    train_run.download_file("outputs/model.pkl", inference_folder + "/model.pkl")

    inference_env = train_run.get_environment()

    config = ScriptRunConfig(
        source_directory=inference_folder,
        script="forecasting_script.py",
        arguments=[
            "--target_column_name",
            target_column_name,
            "--test_dataset",
            test_dataset.as_named_input(test_dataset.name),
        ],
        compute_target=compute_target,
        environment=inference_env,
    )

    run = test_experiment.submit(
        config,
        tags={
            "training_run_id": train_run.id,
            "run_algorithm": train_run.properties["run_algorithm"],
            "valid_score": train_run.properties["score"],
            "primary_metric": train_run.properties["primary_metric"],
        },
    )

    run.log("run_algorithm", run.tags["run_algorithm"])
    return run
```

- Run the test experiment

```
#from run_forecast import run_rolling_forecast

remote_run = run_rolling_forecast(
    test_experiment, compute_target, best_run, test, target_column_name
)
remote_run
remote_run.wait_for_completion(show_output=False)
```

- Create metrics calculation functions

```
import pandas as pd
import numpy as np


def APE(actual, pred):
    """
    Calculate absolute percentage error.
    Returns a vector of APE values with same length as actual/pred.
    """
    return 100 * np.abs((actual - pred) / actual)


def MAPE(actual, pred):
    """
    Calculate mean absolute percentage error.
    Remove NA and values where actual is close to zero
    """
    not_na = ~(np.isnan(actual) | np.isnan(pred))
    not_zero = ~np.isclose(actual, 0.0)
    actual_safe = actual[not_na & not_zero]
    pred_safe = pred[not_na & not_zero]
    return np.mean(APE(actual_safe, pred_safe))
```

- Download the prediction results

```
remote_run.download_file("outputs/predictions.csv", "predictions.csv")
df_all = pd.read_csv("predictions.csv")
```

- display results

```
from azureml.automl.core.shared import constants
from azureml.automl.runtime.shared.score import scoring
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt

# use automl metrics module
scores = scoring.score_regression(
    y_test=df_all[target_column_name],
    y_pred=df_all["predicted"],
    metrics=list(constants.Metric.SCALAR_REGRESSION_SET),
)

print("[Test data scores]\n")
for key, value in scores.items():
    print("{}:   {:.3f}".format(key, value))

# Plot outputs
%matplotlib inline
test_pred = plt.scatter(df_all[target_column_name], df_all["predicted"], color="b")
test_test = plt.scatter(
    df_all[target_column_name], df_all[target_column_name], color="g"
)
plt.legend(
    (test_pred, test_test), ("prediction", "truth"), loc="upper left", fontsize=8
)
plt.show()
```

```
#from metrics_helper import MAPE, APE

df_all.groupby("horizon_origin").apply(
    lambda df: pd.Series(
        {
            "MAPE": MAPE(df[target_column_name], df["predicted"]),
            "RMSE": np.sqrt(
                mean_squared_error(df[target_column_name], df["predicted"])
            ),
            "MAE": mean_absolute_error(df[target_column_name], df["predicted"]),
        }
    )
)
```

```
df_all_APE = df_all.assign(APE=APE(df_all[target_column_name], df_all["predicted"]))
APEs = [
    df_all_APE[df_all["horizon_origin"] == h].APE.values
    for h in range(1, forecast_horizon + 1)
]

%matplotlib inline
plt.boxplot(APEs)
plt.yscale("log")
plt.xlabel("horizon")
plt.ylabel("APE (%)")
plt.title("Absolute Percentage Errors by Forecast Horizon")

plt.show()
```

- End to end Training and testing is completed