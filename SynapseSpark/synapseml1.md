# SynapseML (Open Source) in Azure Syanpse Analtyics Spark

## How to run synalpseml in Azure Syanpse Analtyics Spark

## Prerequsites

- Azure Susbcription
- Azure Syanpse Analtyics Workspace
- SynapseML installation instruction

## Steps

- Create a new Spark 3.1 cluster
- For spark 3.1 this step is important
- Spark configuration below image

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/SynapseSpark/images/synapseml1.jpg "Architecture")

- Create a new notebook
- Select the above cluster
- This has to be in first cluster

```
%%configure -f
{
    "name": "synapseml",
    "conf": {
        "spark.jars.packages": "com.microsoft.azure:synapseml_2.12:0.9.4",
        "spark.jars.repositories": "https://mmlspark.azureedge.net/maven",
        "spark.jars.excludes": "org.scala-lang:scala-reflect,org.apache.spark:spark-tags_2.12,org.scalactic:scalactic_2.12,org.scalatest:scalatest_2.12",
        "spark.yarn.user.classpath.first": "true"
    }
}
```

- now imports

```
from synapse.ml import *
from synapse.ml.cognitive import *
```

```
dataFile = "AdultCensusIncome.csv"
import os, urllib
if not os.path.isfile(dataFile):
    urllib.request.urlretrieve("https://mmlspark.azureedge.net/datasets/" + dataFile, dataFile)
data = spark.createDataFrame(pd.read_csv(dataFile, dtype={" hours-per-week": np.float64}))
data.show(5)
```

```
data = data.select([" education", " marital-status", " hours-per-week", " income"])
train, test = data.randomSplit([0.75, 0.25], seed=123)
```

```
from synapse.ml.train import TrainClassifier
from pyspark.ml.classification import LogisticRegression
model = TrainClassifier(model=LogisticRegression(), labelCol=" income").fit(train)
```

```
from synapse.ml.train import ComputeModelStatistics
prediction = model.transform(test)
metrics = ComputeModelStatistics().transform(prediction)
metrics.select('accuracy').show()
```