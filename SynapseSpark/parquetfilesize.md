# Optimize file size in Spark

## Optimize parquet file size in 1GB chunks for analytics

## pre-requisites

- Azure Account
- Azure synapse workspace
- Azure storage account
- Data set size was 14TB
- Spark node with large 16vcores and 112GB memory with 2 nodes

## Synapse Spark

- First get the data from source

```
from pyspark.sql import SparkSession

# Azure storage access info
blob_account_name = 'xxxxxxx' # replace with your blob name
blob_container_name = 'xxxxxxxx' # replace with your container name
blob_relative_path = '' # replace with your relative folder path
linked_service_name = 'BenchMarkLogs' # replace with your linked service name

#blob_sas_token = mssparkutils.credentials.getConnectionStringOrCreds(linked_service_name)
blob_sas_token = mssparkutils.credentials.getSecret("iamkeys", "benchmarklogs")


# Allow SPARK to access from Blob remotely

wasb_path = 'wasbs://%s@%s.blob.core.windows.net/%s' % (blob_container_name, blob_account_name, blob_relative_path)

spark.conf.set('fs.azure.sas.%s.%s.blob.core.windows.net' % (blob_container_name, blob_account_name), blob_sas_token)
print('Remote blob path: ' + wasb_path)
```

- Can we read the data

```
df = spark.read.parquet(wasb_path)
```

- setup up parquet size

```
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.binSize", "1073741824")
spark.conf.set("spark.sql.files.maxPartitionBytes", "1073741824")
```

- print out the config

```
print(spark.conf.get("spark.microsoft.delta.optimizeWrite.binSize"))
print(spark.conf.get("spark.sql.files.maxPartitionBytes"))
```

- Write back to parquet

```
df.write.mode("overwrite").parquet("abfss://containername@storagename.dfs.core.windows.net/bechmarklogs1/")
```