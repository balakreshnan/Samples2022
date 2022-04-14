# Azure Synapse serverless sql

## use serverless sql to analyze storage logs

## use case

- lets analyze the storage account
- logs are stored in year/month/day type format

## Code

- Make sure the storage has $logs folders
- Logs are stored in $logs/blob/<year>/<month>/<day>/<hour>/<filename>.log

- Create File format

```
DROP EXTERNAL FILE FORMAT [SynapseDelimitedTextFormatlogs];

IF NOT EXISTS (SELECT * FROM sys.external_file_formats WHERE name = 'SynapseDelimitedTextFormatlogs') 
	CREATE EXTERNAL FILE FORMAT [SynapseDelimitedTextFormatlogs] 
	WITH ( FORMAT_TYPE = DELIMITEDTEXT ,
	       FORMAT_OPTIONS (
			 FIELD_TERMINATOR = ';',
			 USE_TYPE_DEFAULT = FALSE,
			 Encoding = 'UTF8',
			 STRING_DELIMITER = '0x22' 
			))
GO
```
- Create data source

```
DROP EXTERNAL DATA SOURCE [logs_accsynapsestorage_dfs_core_windows_net];

IF NOT EXISTS (SELECT * FROM sys.external_data_sources WHERE name = 'logs_accsynapsestorage_dfs_core_windows_net') 
	CREATE EXTERNAL DATA SOURCE [logs_accsynapsestorage_dfs_core_windows_net] 
	WITH (
		LOCATION = 'abfss://$logs@accsynapsestorage.dfs.core.windows.net'
	)
GO
```

- Create table

```
DROP EXTERNAL TABLE logscheck
GO

CREATE EXTERNAL TABLE logscheck
(
	versionno FLOAT,
    trxdate varchar(300) COLLATE Latin1_General_100_BIN2_UTF8,
    svcname varchar(300) COLLATE Latin1_General_100_BIN2_UTF8,
    status VARCHAR(50) COLLATE Latin1_General_100_BIN2_UTF8,
    httpcode INT,
    tvalue INT,
    t2value INT,
    Auth VARCHAR(40) COLLATE Latin1_General_100_BIN2_UTF8,
    storageaccname VARCHAR(100) COLLATE Latin1_General_100_BIN2_UTF8,
    storeaccount VARCHAR(100) COLLATE Latin1_General_100_BIN2_UTF8,
    storetype VARCHAR(50) COLLATE Latin1_General_100_BIN2_UTF8,
    httpurl VARCHAR(4000) COLLATE Latin1_General_100_BIN2_UTF8,
    container VARCHAR(400) COLLATE Latin1_General_100_BIN2_UTF8,
    runid VARCHAR(500) COLLATE Latin1_General_100_BIN2_UTF8,
    size INT,
    ipaddress VARCHAR(300) COLLATE Latin1_General_100_BIN2_UTF8,
    querydate varchar(300) COLLATE Latin1_General_100_BIN2_UTF8,
    t1 INT,
    t2 INT,
    t3 INT,
    t4 INT,
    t5 INT,
    col1 VARCHAR(50) COLLATE Latin1_General_100_BIN2_UTF8,
    col2 VARCHAR(50) COLLATE Latin1_General_100_BIN2_UTF8,
    col3 VARCHAR(50) COLLATE Latin1_General_100_BIN2_UTF8,
    col4 VARCHAR(50) COLLATE Latin1_General_100_BIN2_UTF8,
    col5 VARCHAR(50) COLLATE Latin1_General_100_BIN2_UTF8,
    col6 VARCHAR(4000) COLLATE Latin1_General_100_BIN2_UTF8,
    col7 VARCHAR(50) COLLATE Latin1_General_100_BIN2_UTF8,
    col8 VARCHAR(4000) COLLATE Latin1_General_100_BIN2_UTF8,
    col9 VARCHAR(50) COLLATE Latin1_General_100_BIN2_UTF8,
    col10 VARCHAR(50) COLLATE Latin1_General_100_BIN2_UTF8,
    col11 VARCHAR(50) COLLATE Latin1_General_100_BIN2_UTF8,
    col12 VARCHAR(50) COLLATE Latin1_General_100_BIN2_UTF8,
    col13 VARCHAR(4000) COLLATE Latin1_General_100_BIN2_UTF8,
    col14 VARCHAR(50) COLLATE Latin1_General_100_BIN2_UTF8,
    col15 VARCHAR(50) COLLATE Latin1_General_100_BIN2_UTF8,
    col16 VARCHAR(4000) COLLATE Latin1_General_100_BIN2_UTF8 
)
	WITH (
	LOCATION = 'blob/*/*/*/*/*.log',
	DATA_SOURCE = [logs_accsynapsestorage_dfs_core_windows_net],
	FILE_FORMAT = [SynapseDelimitedTextFormatlogs]
	)
GO
```

- Select data

```
SELECT TOP 100 * FROM dbo.logscheck
GO
```