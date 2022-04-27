# Azure Syanpse and Databricks Delta read and write

## Goal to test delta in Azure synapse analytics spark and azure databricks

## Code

- Azure Synapse Spark Delta

```
%%pyspark
df = spark.read.load('abfss://container@storage.dfs.core.windows.net/*.parquet', format='parquet')
display(df.limit(10))
```

- Now write back to delta

```
df.repartition(10).write.mode("overwrite").format("delta").save('abfss://containername@storage.dfs.core.windows.net/nycgreen')
```

- Now go to Databricks
- Real delta and overwrite

- Read

```
dbutils.fs.mount(
  source = "wasbs://containername@storagename.blob.core.windows.net",
  mount_point = "/mnt/tmpdata",
  extra_configs = {"fs.azure.account.key.storagename.blob.core.windows.net":"xxxxxx"})
```

- Now read the data

```
df = spark.read.format("delta").load("/mnt/tmpdata/nycgreen")
```

- Let write back to same delta location

```
df.repartition(10).write.mode("overwrite").format("delta").save("/mnt/tmpdata/nycgreen")
```

## Done