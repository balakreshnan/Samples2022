# Use SynapseML to process large scale pdf with Form Recognition

## Using Azure Cognitive Services to process large scale pdf with Form Recognition

## Prerequisites

- Azure Account
- Azure Storage account
- Azure Cognitive Services
- Azure synapse analytics
- Create a container and upload the pdf file
- Create a SAS key for the container

## Process using SynapseML and Spark

- Create Spark 3.2 Preview spark pool
- Create a new Notebook and select the new pool created
- Now load the latest synapseml preview for document api processing

```
%%configure -f
{
    "name": "synapseml",
    "conf": {
        "spark.jars.packages": "com.microsoft.azure:synapseml_2.12:0.9.5-96-b25feab9-SNAPSHOT",
        "spark.jars.repositories": "https://mmlspark.azureedge.net/maven",
        "spark.jars.excludes": "org.scala-lang:scala-reflect,org.apache.spark:spark-tags_2.12,org.scalactic:scalactic_2.12,org.scalatest:scalatest_2.12",
        "spark.yarn.user.classpath.first": "true"
    }
}
```

- Lets import the necessary libraries

```
import os
if os.environ.get("AZURE_SERVICE", None) == "Microsoft.ProjectArcadia":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
```

- Now the regular libraries

```
from pyspark.sql.functions import udf, col
from synapse.ml.io.http import HTTPTransformer, http_udf
from requests import Request
from pyspark.sql.functions import lit
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
import os
```

- Lets import synapseml

```
from synapse.ml.cognitive import *
```

- Now lets read few images and test

```
from pyspark.sql.functions import col, explode

# Create a dataframe containing the source files
imageDf = spark.createDataFrame([
  ("https://storageaccount.dfs.core.windows.net/containername/billoflading/billofladding1.png?sp=r&st=2022xxxxxx",
  "https://storageaccount.dfs.core.windows.net/containername/billoflading/billofladding2.png?sp=r&st=2022xxxxx",
  "https://storageaccount.dfs.core.windows.net/containername/billoflading/BillofLading_Labeled_resized.jpg?sp=r&st=2022xxxxx",)
], ["source",])

# Run the Form Recognizer service
analyzeLayouts = (AnalyzeDocument()
                 .setSubscriptionKey("xxxxxxxxxxxxxxxxxxxxxxxxxxx")
                 .setLocation("eastus2")
                 .setPrebuiltModelId("prebuilt-document")
                 .setImageUrlCol("source")
                 .setOutputCol("Layouts"))
# Show the results of recognition.
display(analyzeLayouts
        .transform(imageDf)
        .withColumn("documentsresult", col("Layouts.analyzeResult"))
        .select("source", "documentsresult"))
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/SynapseSpark/images/synapsemlforms4.jpg "Architecture")

- Pull only keyvaulr pairs

```
from pyspark.sql.functions import col, explode

# Create a dataframe containing the source files
imageDf = spark.createDataFrame([
  ("https://storageaccount.dfs.core.windows.net/containername/billoflading/billofladding1.png?sp=r&st=2022xxxxxx",
  "https://storageaccount.dfs.core.windows.net/containername/billoflading/billofladding2.png?sp=r&st=2022xxxxx",
  "https://storageaccount.dfs.core.windows.net/containername/billoflading/BillofLading_Labeled_resized.jpg?sp=r&st=2022xxxxx",)
], ["source",])

# Run the Form Recognizer service
analyzeLayouts = (AnalyzeDocument()
                 .setSubscriptionKey("xxxxxxxxxxxxxxxxxxxxxxxxxxx")
                 .setLocation("eastus2")
                 .setPrebuiltModelId("prebuilt-document")
                 .setImageUrlCol("source")
                 .setOutputCol("Layouts"))
# Show the results of recognition.
display(analyzeLayouts
        .transform(imageDf)
        .withColumn("documentsresult", col("Layouts.analyzeResult.keyValuePairs"))
        .select("source", "documentsresult"))
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/SynapseSpark/images/synapsemlforms3.jpg "Architecture")

- Show only entities

```
# Show the results of recognition.
display(analyzeLayouts
        .transform(imageDf)
        .withColumn("documentsresult", col("Layouts.analyzeResult.entities"))
        .select("source", "documentsresult"))
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/SynapseSpark/images/synapsemlforms2.jpg "Architecture")

- Now only tables

```
# Show the results of recognition.
display(analyzeLayouts
        .transform(imageDf)
        .withColumn("documentsresult", col("Layouts.analyzeResult.tables"))
        .select("source", "documentsresult"))
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/SynapseSpark/images/synapsemlforms1.jpg "Architecture")

## AnalyzeLayout

- Using Layout api

```
from pyspark.sql.functions import col, explode

# Create a dataframe containing the source files
imageDf = spark.createDataFrame([
  ("https://storagename.dfs.core.windows.net/containername/billoflading/billofladding1.png?sp=r&st=2022xxxx",
  "https://storagename.dfs.core.windows.net/containername/billoflading/billofladding2.png?sp=r&st=2022xxxx",
  "https://storagename.dfs.core.windows.net/containername/billoflading/BillofLading_Labeled_resized.jpg?sp=r&st=2022xxx",)
], ["source",])

# Run the Form Recognizer service
analyzeLayouts = (AnalyzeLayout()
                 .setSubscriptionKey("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                 .setLocation("westus2")
                 .setImageUrlCol("source")
                 .setOutputCol("Layouts"))
# Show the results of recognition.
display(analyzeLayouts
        .transform(imageDf)
        .withColumn("documentsresult", explode(col("Layouts.analyzeResult.readResults")))
        .select("source", "documentsresult"))
```

- Now display page results

```
from pyspark.sql.functions import col, explode

# Create a dataframe containing the source files
imageDf = spark.createDataFrame([
  ("https://storagename.dfs.core.windows.net/containername/billoflading/billofladding1.png?sp=r&st=2022xxxx",
  "https://storagename.dfs.core.windows.net/containername/billoflading/billofladding2.png?sp=r&st=2022xxxx",
  "https://storagename.dfs.core.windows.net/containername/billoflading/BillofLading_Labeled_resized.jpg?sp=r&st=2022xxx",)
], ["source",])

# Run the Form Recognizer service
analyzeLayouts = (AnalyzeLayout()
                 .setSubscriptionKey("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                 .setLocation("westus2")
                 .setImageUrlCol("source")
                 .setOutputCol("Layouts"))
# Show the results of recognition.
display(analyzeLayouts
        .transform(imageDf)
        .withColumn("documentsresult", explode(col("Layouts.analyzeResult.pageResults")))
        .select("source", "documentsresult"))
```