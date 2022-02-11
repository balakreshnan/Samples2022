# End to End Data Flow with transformation performance testing

## Run Data transformation with dataflow with 1.5 billion rows dataset

## Goal

- Run Data transformation with dataflow with 1.5 billion rows dataset
- Read parquet and do transformation and save as delta
- Use as is dataflow configuration
- Use Memory optimized dataflow configuration for spark
- Use Standard dataflow configuration for spark
- Data set is nyc taxi data (yelllow cab)
- Open source data set and no PII or sensitive data

## Prepare data

- First get the NYC taxi yellow data.
- I have it download as parquet
- Synapse analytics has sample dataset for the above and you can save it as parquet

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow2.jpg "Entire Flow")

- Above shows the list of files and size
- now lets valdiate the data
- Open Serverless SQL query
- Run the below query to get row count

```
select count(*) 
FROM
    OPENROWSET(
        BULK     'https://storagename.blob.core.windows.net/rootfolder/nyctaxiyellow/*',
        FORMAT = 'parquet'
    ) AS [result] 
```

- Now for the output

```
1571671152
```

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow1.jpg "Entire Flow")

## Data Flow with Data flow script

- Next we are going to create a simple flow
- First create a pipeline and then add 2 activities
- One to delete old files in destination folder
- Next would be a data flow task

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow3.jpg "Entire Flow")

- Here is the details from delete activity

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow4.jpg "Entire Flow")

- Next lets dig deep into Data flow
- Create a new dataflow task
- Below is the high level flow

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow5.jpg "Entire Flow")

- Source is going to be the nyctaxiyellow folder with parquet files from above section
- COnfiguration for sources

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow6.jpg "Entire Flow")

- here is the configuration for linked services

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow7.jpg "Entire Flow")

- Next do some transformation
- Let's do Distinct rows

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow8.jpg "Entire Flow")

- Now here is the data flow script used

```
source1 aggregate(groupBy(mycols = sha2(256,columns())),
	each(match(true()), $$ = first($$))) ~> DistinctRows
```

- Now lets see if we can check for nulls
- Has nulls and no nulls will have 2 different path

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow9.jpg "Entire Flow")

- Data flow script code 

```
LookForNULLs@hasNULLs select(mapColumn(
		mycols,
		vendorID,
		tpepPickupDateTime,
		tpepDropoffDateTime,
		passengerCount,
		tripDistance,
		puLocationId,
		doLocationId,
		startLon,
		startLat,
		endLon,
		endLat,
		rateCodeId,
		storeAndFwdFlag,
		paymentType,
		fareAmount,
		extra,
		mtaTax,
		improvementSurcharge,
		tipAmount,
		tollsAmount,
		totalAmount,
		puYear,
		puMonth
	),
	skipDuplicateMapInputs: true,
	skipDuplicateMapOutputs: true) ~> Select2
```

- Now for non nulls
- Data flow code

```
LookForNULLs@noNULLs select(mapColumn(
		vendorID,
		tpepPickupDateTime,
		tpepDropoffDateTime,
		passengerCount,
		tripDistance,
		puLocationId,
		doLocationId,
		startLon,
		startLat,
		endLon,
		endLat,
		rateCodeId,
		storeAndFwdFlag,
		paymentType,
		fareAmount,
		extra,
		mtaTax,
		improvementSurcharge,
		tipAmount,
		tollsAmount,
		totalAmount,
		puYear,
		puMonth
	),
	skipDuplicateMapInputs: true,
	skipDuplicateMapOutputs: true) ~> Select1
```

- Now let's select the above 2 path outputs and sink to destination folder with delta

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow10.jpg "Entire Flow")

- Now sink the data to destination folder
- we are using delta format
- Sink actual rows with distinct rows and no nulls

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow11.jpg "Entire Flow")

- Here is the paritioning

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow12.jpg "Entire Flow")

- folder for delta

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow13.jpg "Entire Flow")

- Sink rows that has nulls to different folder in delta format

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow14.jpg "Entire Flow")

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow15.jpg "Entire Flow")

- Now save the data flow
- Commit the pipeline
- If connected to git then create pull request and merge to main branch
- Switch the branch to main and then publish
- then switch back to the WIP branch
- Click Trigger Now and start the flow
- Go to monitor section and view the pipeline runs
- List of runs

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow16.jpg "Entire Flow")

- Look into the run details for 32+16 cores

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow17.jpg "Entire Flow")

- Lineage through purview

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow18.jpg "Entire Flow")

- Run for 16+16 Cores
- In progress spark application runs

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow19.jpg "Entire Flow")

- Details graph on tasks

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow20.jpg "Entire Flow")

- Took more than 2 hours

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow21.jpg "Entire Flow")

- Details of the run

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow23.jpg "Entire Flow")

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow22.jpg "Entire Flow")

- cost

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow24.jpg "Entire Flow")

- Now 32+16 Cores

- Entire Flow

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow27.jpg "Entire Flow")

- Pipeline run details and progress

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow26.jpg "Entire Flow")

- Spark UI with tasks

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow25.jpg "Entire Flow")

- Completed run in 55 mins

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow28.jpg "Entire Flow")

- Details on each tasks

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/nyctaxiyellow29.jpg "Entire Flow")

## Done