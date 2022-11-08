# Azure Machine learning managed spark

## Managed Spark introduction using Azure Machine learning services

## Prerequisites

- Azure Account
- Azure machine learning service
- Titanic dataset


## Managed Spark Steps

- Fist go to Notebooks
- Create a new notebook
- Next to Compute selection click new and select AzureML spark compute

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/managedspark1.jpg "Architecture")

- Now back to notebook and you should see the compute created. It usually takes 3-5 minutes

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/managedspark2.jpg "Architecture")

- If you want to change session click configure session in the botton left of notebook

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/managedspark3.jpg "Architecture")

- by default when i created it was spark 3.2

- Now lets load some data from external file
- Below was the open source available data set

```
df = spark.read.option("header", "true").csv("wasbs://demo@dprepdata.blob.core.windows.net/Titanic.csv")
```

- Let's do some data processing

```
from pyspark.sql.functions import col, desc

df.filter(col('Survived') == 1).groupBy('Age').count().orderBy(desc('count')).show(10)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/managedspark4.jpg "Architecture")

- Now lets load from Azure machine learning data set

```
import azureml.core
print(azureml.core.VERSION)

from azureml.core import Workspace, Dataset
ws = Workspace.get(name='workspacename', subscription_id='xxxxxxx', resource_group='rgname')
ds = Dataset.get_by_name(ws, "titanic")
df = ds.to_spark_dataframe()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/managedspark5.jpg "Architecture")

## Machine learning

- Lets now build a simple machine learning model

```
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import mean,col,split, col, regexp_extract, when, lit
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer
```

- Lets see the stats for dataframe

```
df.describe().show()
```

- Display the schema

```
df.printSchema()
```

- Let's do some data engineering

```
df.select("Survived","Pclass","Embarked").show()
```

- group by and count

```
df.groupBy("Sex","Survived").count().show()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/managedspark6.jpg "Architecture")

- Now fill NA values

```
df = df.na.fill({"Embarked" : 'S'})
df = df.drop("Cabin")
```

- Create new columns values

```
df = df.withColumn("Family_Size",col('SibSp')+col('Parch'))
```

- Index the values

```
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in ["Sex","Embarked"]]
pipeline = Pipeline(stages=indexers)
df = pipeline.fit(df).transform(df)
```

- Drop unncessary columns

```
df = df.drop("PassengerId","Name","Ticket","Cabin","Embarked","Sex")
```

- fill null values

```
df = df.na.fill({"Age" : 10})
```

- Change column to string to double

```
#Using withColumn() examples
df = df.withColumn("Age",df.Age.cast('double'))
df = df.withColumn("SibSp",df.SibSp.cast('double'))
df = df.withColumn("Parch",df.Parch.cast('double'))
df = df.withColumn("Fare",df.Fare.cast('double'))
df = df.withColumn("Pclass",df.Pclass.cast('double'))
df = df.withColumn("Survived",df.Survived.cast('double'))
```

- Featurizer
- 
```
feature = VectorAssembler(inputCols=df.columns[1:],outputCol="features")
feature_vector= feature.transform(df)
```

- Split train and test data

```
(trainingData, testData) = feature_vector.randomSplit([0.8, 0.2],seed = 11)
```

- Logistic regression

```
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="Survived", featuresCol="features")
#Training algo
lrModel = lr.fit(trainingData)
lr_prediction = lrModel.transform(testData)
lr_prediction.select("prediction", "Survived", "features").show()
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
```

- metric calculation

```
lr_accuracy = evaluator.evaluate(lr_prediction)
print("Accuracy of LogisticRegression is = %g"% (lr_accuracy))
print("Test Error of LogisticRegression = %g " % (1.0 - lr_accuracy))
```

- Now Decistion Tree

```
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="Survived", featuresCol="features")
dt_model = dt.fit(trainingData)
dt_prediction = dt_model.transform(testData)
dt_prediction.select("prediction", "Survived", "features").show()
```

```
dt_accuracy = evaluator.evaluate(dt_prediction)
print("Accuracy of DecisionTreeClassifier is = %g"% (dt_accuracy))
print("Test Error of DecisionTreeClassifier = %g " % (1.0 - dt_accuracy))
```

- another model for Gradient Boosted Tree

```
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(labelCol="Survived", featuresCol="features",maxIter=10)
gbt_model = gbt.fit(trainingData)
gbt_prediction = gbt_model.transform(testData)
gbt_prediction.select("prediction", "Survived", "features").show()
```

- Metric calculation

```
gbt_accuracy = evaluator.evaluate(gbt_prediction)
print("Accuracy of Gradient-boosted tree classifie is = %g"% (gbt_accuracy))
print("Test Error of Gradient-boosted tree classifie %g"% (1.0 - gbt_accuracy))
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/managedspark7.jpg "Architecture")