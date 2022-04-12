# Azure Databricks TPCH Test with external Storage

## TPCH test with external storage to analyze Storage Read and Writes

## Prerequisites

- Azure Account
- Azure Storage
- Azure Databricks
- Azure data factory to copy TPCH data to Azure Storage
- Total rows in LineItem is 60 Billion
- Create a service principal for Azure databricks to access storage
- provide Storage blob data reader to service principal

## Steps

- Make sure Storage Blob data reader permission is applied to service principal for Storage
- Double check the TPCH data is copied to Azure Storage
- First lets set the service principal for Azure Databricks to access storage
- Replace storagename with your storage account name
- Get the service principal client id, secret, and tenant id for the below

```
%scala
spark.conf.set("fs.azure.account.auth.type.storagename.dfs.core.windows.net", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type.storagename.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id.storagename.dfs.core.windows.net", "xxxxxxxxxxxxxxxxxxxxxx")
spark.conf.set("fs.azure.account.oauth2.client.secret.storagename.dfs.core.windows.net", "xxxxxxxxxxxxxxxxxxxxxxxxx")
spark.conf.set("fs.azure.account.oauth2.client.endpoint.storagename.dfs.core.windows.net", "https://login.microsoftonline.com/xxxxxxxxxxxxxxxxx/oauth2/token")
```

- Now lets load all the data from Azure Storage
- First customer

```
customerdf = spark.read.parquet("abfss://containername@storageaccountname.dfs.core.windows.net/CUSTOMER")
```

- Create a customer spark sql

```
%sql
CREATE TEMPORARY VIEW customer
USING org.apache.spark.sql.parquet
OPTIONS (
  path "abfss://containername@storageaccountname.dfs.core.windows.net/CUSTOMER/*.parquet"
)
```

- Repeat this for the other tables
- Now line item

```
lineitem = spark.read.parquet("abfss://containername@storageaccountname.dfs.core.windows.net/LINEITEM/*.parquet")
```

```
%sql
CREATE TEMPORARY VIEW lineitem
USING org.apache.spark.sql.parquet
OPTIONS (
  path "abfss://containername@storageaccountname.dfs.core.windows.net/LINEITEM/*.parquet"
)
```

- Nation

```
nation = spark.read.parquet("abfss://containername@storageaccountname.dfs.core.windows.net/NATION/*.parquet")
```

```
%sql
CREATE TEMPORARY VIEW nation
USING org.apache.spark.sql.parquet
OPTIONS (
  path "abfss://containername@storageaccountname.dfs.core.windows.net/NATION/*.parquet"
)
```

- Orders

```
orders = spark.read.parquet("abfss://containername@storageaccountname.dfs.core.windows.net/ORDERS/*.parquet")
```

```
%sql
CREATE TEMPORARY VIEW orders
USING org.apache.spark.sql.parquet
OPTIONS (
  path "abfss://containername@storageaccountname.dfs.core.windows.net/ORDERS/*.parquet"
)
```

- Parts

```
part = spark.read.parquet("abfss://containername@storageaccountname.dfs.core.windows.net/PART/*.parquet")
```

```
%sql
CREATE TEMPORARY VIEW part
USING org.apache.spark.sql.parquet
OPTIONS (
  path "abfss://containername@storageaccountname.dfs.core.windows.net/PART/*.parquet"
)
```

- Part Supplier

```
partsupp = spark.read.parquet("abfss://containername@storageaccountname.dfs.core.windows.net/PARTSUPP/*.parquet")
```

```
%sql
CREATE TEMPORARY VIEW partsupp
USING org.apache.spark.sql.parquet
OPTIONS (
  path "abfss://containername@storageaccountname.dfs.core.windows.net/PARTSUPP/*.parquet"
)
```

- Region

```
region = spark.read.parquet("abfss://containername@storageaccountname.dfs.core.windows.net/REGION/*.parquet")
```

```
%sql
CREATE TEMPORARY VIEW region
USING org.apache.spark.sql.parquet
OPTIONS (
  path "abfss://containername@storageaccountname.dfs.core.windows.net/REGION/*.parquet"
)
```

- Supplier

```
supplier = spark.read.parquet("abfss://containername@storageaccountname.dfs.core.windows.net/SUPPLIER/*.parquet")
```

```
%sql
CREATE TEMPORARY VIEW supplier
USING org.apache.spark.sql.parquet
OPTIONS (
  path "abfss://containername@storageaccountname.dfs.core.windows.net/SUPPLIER/*.parquet"
)
```

```
lineitem.count()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/tpch1.jpg "Architecture")

- write

```
dfout = spark.sql('''select
       l_returnflag,
       l_linestatus,
       sum(l_quantity) as sum_qty,
       sum(l_extendedprice) as sum_base_price,
       sum(l_extendedprice * (1-l_discount)) as sum_disc_price,
       sum(l_extendedprice * (1-l_discount) * (1+l_tax)) as sum_charge,
       avg(l_quantity) as avg_qty,
       avg(l_extendedprice) as avg_price,
       avg(l_discount) as avg_disc,
       count(*) as count_order
 from
       lineitem
 group by
       l_returnflag,
       l_linestatus
 order by
       l_returnflag;''')
```

```
dfout.repartition(20).write.parquet('abfss://containername@storageaccountname.dfs.core.windows.net/tpschoutput1/')
```