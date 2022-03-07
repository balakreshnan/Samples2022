# Query Azure Synapse Serverless SQL using Azure Databricks

## Query Azure Synapse Serverless SQL using Azure Databricks

## Requirements

- Azure Account
- Azure Storage
- Azure Databricks
- Azure Synapse Analytics workspace
- Load the NYC taxi data

## Azure Synapse Serverless SQL Setup

- Lets create a Database and setup up Serverless SQL

```
Create database svrsqldb;
CREATE MASTER KEY ENCRYPTION BY PASSWORD = 'xxxxxxx';
```

- Now create a credential with managed identity

```
CREATE DATABASE SCOPED CREDENTIAL SynapseIdentity
WITH IDENTITY = 'Managed Identity';
GO
```

- Create a data source for us to use in the view

```
CREATE EXTERNAL DATA SOURCE synroot
WITH (    LOCATION   = 'https://storageaccount.dfs.core.windows.net/containerroot/',
          CREDENTIAL = SynapseIdentity
)
```

- Now create a view

```
CREATE VIEW vwYellowTaxi
AS SELECT *
FROM
    OPENROWSET(
        BULK 'nyctaxiyellow/*',
        DATA_SOURCE = 'synroot',
        FORMAT='PARQUET'
    ) AS nyc
GO
```

- View can be parquet or delta

- Now to secure the view by granting user access
- to provide access to Azure AD account
- User has to have login first
- Then provide select on the view created
- View to data store is accessed using managed identity
- Other option is to do pass through authentication in this case make sure the user id has storage blob data reader to the ADLS gen2 account used

```
CREATE USER [mike@contoso.com] FROM EXTERNAL PROVIDER;

GRANT SELECT ON OBJECT::dbo.vwYellowTaxi TO [mike@contoso.com];
```

## Query Azure Synapse Serverless SQL using Azure Databricks

- Log into Azure databricks
- Create a new cluster with latest runtime
- Create a new python notebook
- First load the sql driver

```
%scala

Class.forName("com.microsoft.sqlserver.jdbc.SQLServerDriver")
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbsvrsql1.jpg "Architecture")

- Now set the parameters for connection to Azure Syanpse Serverless SQL using JDBC

```
jdbcHostname = "sqlname-ondemand.sql.azuresynapse.net"
jdbcDatabase = "dbname"
jdbcPort = 1433
jdbcUrl = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname, jdbcPort, jdbcDatabase)
connectionProperties = {
"user" : "user",
"password" : "password",
"driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}
```

- Next read some data

```
Spdf = spark.read.jdbc(url=jdbcUrl, table="(select top 100 * from dbo.vwYellowTaxi) count", properties=connectionProperties).limit(100)
display(Spdf)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbsvrsql2.jpg "Architecture")