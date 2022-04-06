# Access Storage logs using serverless sql

## Parse $logs folder with Serverless SQL

## Prerequisties

- Azure Storage
- Enable diagnostics logs
- Azure synapse analytics
- Create a serverless sql query
- Create a database

## SQL Code

- Make sure linked service is created with permission to the storage account
- write sql

```
SELECT
    Top 100 * 
FROM
    OPENROWSET(
        BULK 'https://storageaccname.dfs.core.windows.net/$logs/blob/2022/*/*/*/*.log',
        FORMAT = 'CSV',
        PARSER_VERSION = '2.0',
        FIELDTERMINATOR =';', 
        ROWTERMINATOR = '0x0a', 
        FIELDQUOTE = '"'
    )
    WITH (
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
    ) AS row;
```
