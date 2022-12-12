# Azure Machine learning - Ray parallel processing

## Execute Ray parallel processing in Azure Machine Learning with single node CPU

## Prerequisites

- Azure subscription
- Azure Machine Learning Workspace
- Azure Storage Account
- Install Ray library

## Code

- install ray libraries

```
pip install -qU "ray[tune]" lightgbm_ray
```

- install pyarrow specific version for the code to work

```
pip install pyarrow==6.0.1
```

- Restart the kernel
- import libraries

```
from typing import Tuple

import ray
from ray.train.batch_predictor import BatchPredictor
from ray.train.lightgbm import LightGBMPredictor
from ray.data.preprocessors.chain import Chain
from ray.data.preprocessors.encoder import Categorizer
from ray.train.lightgbm import LightGBMTrainer
from ray.air.config import ScalingConfig
from ray.data.dataset import Dataset
from ray.air.result import Result
from ray.data.preprocessors import StandardScaler
```

- Now prepare the data function

```
def prepare_data() -> Tuple[Dataset, Dataset, Dataset]:
    dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer_with_categorical.csv")
    train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)
    test_dataset = valid_dataset.drop_columns(cols=["target"])
    return train_dataset, valid_dataset, test_dataset
```

- Now create the train function

```
def train_lightgbm(num_workers: int, use_gpu: bool = False) -> Result:
    train_dataset, valid_dataset, _ = prepare_data()

    # Scale some random columns, and categorify the categorical_column,
    # allowing LightGBM to use its built-in categorical feature support
    preprocessor = Chain(
        Categorizer(["categorical_column"]), 
        StandardScaler(columns=["mean radius", "mean texture"])
    )

    # LightGBM specific params
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "binary_error"],
    }

    trainer = LightGBMTrainer(
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        label_column="target",
        params=params,
        datasets={"train": train_dataset, "valid": valid_dataset},
        preprocessor=preprocessor,
        num_boost_round=100,
    )
    result = trainer.fit()
    print(result.metrics)

    return result
```

- Now create the predict function

```
def predict_lightgbm(result: Result):
    _, _, test_dataset = prepare_data()
    batch_predictor = BatchPredictor.from_checkpoint(
        result.checkpoint, LightGBMPredictor
    )

    predicted_labels = (
        batch_predictor.predict(test_dataset)
        .map_batches(lambda df: (df > 0.5).astype(int), batch_format="pandas")
    )
    print(f"PREDICTED LABELS")
    predicted_labels.show()

    shap_values = batch_predictor.predict(test_dataset, pred_contrib=True)
    print(f"SHAP VALUES")
    shap_values.show()
```

- now lets invoke the training
- i have 16 cpu cores in my machine

```
result = train_lightgbm(num_workers=7, use_gpu=False)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/ray1.jpg "Architecture")

- now predict the results

```
predict_lightgbm(result)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/ray2.jpg "Architecture")