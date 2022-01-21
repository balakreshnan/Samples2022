# Azure Storage Usuage Cost Estimation for Size and Transcation Cost

## Using Azure Synapse Analytics Spark to Estimate the Cost of Storage and Transcation

## Prerequsites

- Azure Susbcription
- Azure Syanpse Analtyics Workspace
- Azure Storage Account used
- Create a Spark Cluster

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/SynapseSpark/images/synapseml4.jpg "Architecture")

## Concepts

- Run every month to get blob sizes
- Run Transcation cost estimation
- Save back to Azure Storage for further analysis

## What needs to be enabled

- Storage account should have diagnostics logs enabled to store in $logs folder
- Create a Blob Inventory for the storage account and schedule it to run every month to calculate Size
- Transcation cost should are captures in $logs folder

## Code

- Lets configure some parameters and URL for inventory data pull
- I am keeping Year and Month as parameters as all the logs are stored on year/month/day basis
- Easy to read month of data

```
year = "2022"
month= "01"
invurl = "abfss://containername@storageaccountname.dfs.core.windows.net/" + year + "/" + month + "/*/*/DefaultRule-AllBlobs/DefaultRule-AllBlobs.csv"
```

- As you can see DefaultRule is the name and can be configured to store as CSV or parquet
- now the transcation details

```
logurl = "abfss://$logs@storageaccount.dfs.core.windows.net/blob/" + year + "/" + month + "/*/*/*.log"
```

- Now import the libraries

```
import pyspark.sql.functions as f
from pyspark.sql.types import IntegerType,BooleanType,DateType
```

- Read the Inventory data first

```
%%pyspark
df = spark.read.load(invurl, format='csv'
## If header exists uncomment line below
, header=True
)
display(df.limit(10))
```

- Now read the transcation data from $logs folder

```
%%pyspark
dflogs = spark.read.option("delimiter", ";").load(logurl, format='csv'
## If header exists uncomment line below
#, header=True
)
display(dflogs.limit(10))
```

- Concert the Content-Length to integer

```
df = df.withColumn("size",df['Content-Length'].cast('int'))
```

- Lets get the container name

```
split_col = f.split(df['Name'], '/')
df = df.withColumn('ContainerName', split_col.getItem(0))
```

- Aggregate by container name and get size summation

```
dfsize = df.groupBy("ContainerName").sum("size")
```

- Calculate the size first
- Then we calculate approximate pricing
- Size in bytes so we convert to GB

```
CONSTANT = (1024.0 * 1024 * 1024)
dfsize = dfsize.withColumn("TotalGB", (dfsize['sum(size)'] / CONSTANT))
```

- Now pricing
- Cost is from Azure retail pricing and only approximation. (note: doesn't contain any discount)

```
prize = (0.02)
dfsize = dfsize.withColumn("TotalCost$", (dfsize['TotalGB'] * prize))
```

- For $logs there is no columns available so we create columns names

```
columns = ['column1','column2','column3','column4','column5','column6', 'column7']
```

```
from pyspark.sql.functions import col

col_rename = {f'_c{i}':columns[i] for i in range(0,len(columns))}
df_with_col_renamed = dflogs.select([col(c).alias(col_rename.get(c,c)) for c in dflogs.columns])
display(df_with_col_renamed)
```

- Lets get the container name

```
split_col = f.split(df_with_col_renamed['_c12'], '/')
df_with_col_renamed = df_with_col_renamed.withColumn('ContainerName1', split_col.getItem(2))
```

- Also get the year and month from transaction logs

```
split_col = f.split(df_with_col_renamed['column2'], '-')
df_with_col_renamed = df_with_col_renamed.withColumn('year', split_col.getItem(0))
df_with_col_renamed = df_with_col_renamed.withColumn('month', split_col.getItem(1))
```

- Now we are calculate transcation cost

```
dftrans = df_with_col_renamed.groupBy(["ContainerName1", "year", "month"]).count()
```

- Lets Join both the dataframes

```
dfjoin = dfsize.join(dftrans,dfsize.ContainerName ==  dftrans.ContainerName1,"Left")
```

- Calculate price or cost for transcation

```
prizetrans = (0.005)
dfjoin = dfjoin.withColumn("TotalCostTrans$", (dfjoin['count'] * prizetrans))
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/SynapseSpark/images/synapseml2.jpg "Architecture")

- Now lets write back to storage for further analysis

```
outputurl = "abfss://containername@storageacfctname.dfs.core.windows.net/blobbillingoutput/" + year + "/" + month + "/blobbilling" + year + month + ".csv"
```

- Lets write the dataframe to output

```
dfjoin.select("ContainerName", "sum(size)", "TotalGB", "TotalCost$", "year", "month", "count", "TotalCostTrans$").repartition(1).write.option("header", "true").csv(outputurl)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/SynapseSpark/images/synapseml3.jpg "Architecture")

- Spark logs to see how it executed

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/SynapseSpark/images/synapseml5.jpg "Architecture")

- We can also see the spark history server

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/SynapseSpark/images/synapseml6.jpg "Architecture")