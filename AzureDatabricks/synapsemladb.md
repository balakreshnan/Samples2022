# SynapseML (Open Source) in Azure Databricks

## How to run synalpseml in Azure Databricks

## Prerequsites

- Azure Susbcription
- Azure Databricks
- SynapseML installation instruction

## Steps

- First Create a Azure data bricks Workspace
- Create a new Compute
- I tested with Photon and regular spark engines

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/synapseml1.jpg "Architecture")

- Install Library from Maven location
- Here is the cluster configuration tested

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/synapseml2.jpg "Architecture")

- Config for synapseml library installation
- Check the repo URL, it's different

```
maven cordinates: com.microsoft.azure:synapseml_2.12:0.9.5
Repository: https://mmlspark.azureedge.net/maven
```

- Next start the cluster
- Wait for cluster to be running state and also libraries installed
- Next let's create a new notebook

## Code

- Now lets write a sample code
- I am using Git hub repository connected with Repos in Azure databricks

```
import synapse.ml
from synapse.ml.cognitive import *
```

```
import numpy as np
import pandas as pd
```

- now let's load the data

```
dataFile = "AdultCensusIncome.csv"
import os, urllib
if not os.path.isfile(dataFile):
    urllib.request.urlretrieve("https://mmlspark.azureedge.net/datasets/" + dataFile, dataFile)
data = spark.createDataFrame(pd.read_csv(dataFile, dtype={" hours-per-week": np.float64}))
data.show(5)
```

- Select the columns and split for training and test

```
data = data.select([" education", " marital-status", " hours-per-week", " income"])
train, test = data.randomSplit([0.75, 0.25], seed=123)
```

- now train the model
```
from synapse.ml.train import TrainClassifier
from pyspark.ml.classification import LogisticRegression
model = TrainClassifier(model=LogisticRegression(), labelCol=" income").fit(train)
```

- now metric calculation

```
from synapse.ml.train import ComputeModelStatistics
prediction = model.transform(test)
metrics = ComputeModelStatistics().transform(prediction)
metrics.select('accuracy').show()
```

- The above is to show the functionality of the library
- Not intended to solve any use case
- All the code are based on documentation