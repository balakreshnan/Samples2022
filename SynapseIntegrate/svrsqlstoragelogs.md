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
        BULK 'https://storageaccname.dfs.core.windows.net/$logs/blob/*/*/*/*/*.log',
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

- Now create a view

```
Drop view vwStorageLogs;
Create view vwStorageLogs
AS (
    SELECT
     * 
FROM
    OPENROWSET(
        BULK 'https://accsynapsestorage.dfs.core.windows.net/$logs/blob/*/*/*/*/*.log',
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
    ) AS row
)
```

- Now query to make sure we can access the view

```
Select top 100 * from vwStorageLogs;
```

- to cast date

```
Select top 10 svcname, storeaccount, container, ipaddress
, year(CAST(trxdate AS DATETIME2)) as year
, MONTH(CAST(trxdate AS DATETIME2)) as month
, DAY(CAST(trxdate AS DATETIME2)) as day
, col14 as loginuser
from vwStorageLogs;
```

- group by

```
Select svcname, storeaccount, ipaddress
, year(CAST(trxdate AS DATETIME2)) as year
, MONTH(CAST(trxdate AS DATETIME2)) as month
, DAY(CAST(trxdate AS DATETIME2)) as day
, col14 as loginuser
from vwStorageLogs
Group by svcname, storeaccount, ipaddress, col14,
 year(CAST(trxdate AS DATETIME2)), MONTH(CAST(trxdate AS DATETIME2)), 
DAY(CAST(trxdate AS DATETIME2))
Order by svcname, storeaccount, ipaddress, col14,
 year(CAST(trxdate AS DATETIME2)), MONTH(CAST(trxdate AS DATETIME2)), 
DAY(CAST(trxdate AS DATETIME2));
```

- Took 1:25 mins to run the above query again 19 million rows

- doing total count
- Took 41 seconds in serverless sql
- close to 19 million rows

```
select count(*) from vwStorageLogs;
```

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/storagelogs1.jpg "Entire Flow")