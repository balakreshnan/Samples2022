# Azure Synapse Integrate dataflow with Serverless SQL as source

## Move and process data from server less sql and dave as delta

## requirements

- Azure Account
- Azure Storage
- Azure Synapse Analytics workspace
- Load the NYC taxi data

## Steps

- Log into Azure synapse analytics workspace
- Go to Develop on the left menu
- create a new data flow

### Data flow

- Overall data flow

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/svrlesssql9.jpg "Entire Flow")

- First connect to serverless sql using Azure sql database connector
- Get the ondemand sql name from azure synapse workspace overview page
- I am using sql authentication to test Managed identity authentication for serverless sql to storage
- You can also use passthrough authentication, but make sure proper permission is set in the storage account to read the data

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/svrlesssql1.jpg "Entire Flow")

- Select the table name as vwYellowTaxi, in my case it's a view to call

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/svrlesssql2.jpg "Entire Flow")

- Click browse and see if you can see the data

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/svrlesssql3.jpg "Entire Flow")

- Next drag and drop select

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/svrlesssql4.jpg "Entire Flow")

- Select all columns
- Next we are going to create parition columns as year and month
- Drag and drop derived column1 task

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/svrlesssql5.jpg "Entire Flow")

- create new column for year and month

```
year(tpepPickupDateTime)
month(tpepPickupDateTime)
```

- Next is save the file as delta into Synapse workspace storage
- drag sink and select inline
- Select delta and see the config below

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/svrlesssql6.jpg "Entire Flow")

- Specify the storage folder to store the file

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/svrlesssql7.jpg "Entire Flow")

- Now set the partition columns

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/svrlesssql8.jpg "Entire Flow")

- Commit and save the data flow

### Pipeline to run the dataflow

- Now lets create a pipeline
- Go to Pipeline and create new pipeline
- Then drag and drop dataflow activity and select the above dataflow created

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/svrlesssql10.jpg "Entire Flow")

- Set the compute options to 32 + 16 cores
- Select memory optimized
- Commit and publish the pipeline
- Click Add Triger and select Run now
- Wait for the pipeline to run and complete

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/svrlesssql11.jpg "Entire Flow")

- Click details

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/svrlesssql12.jpg "Entire Flow")

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/svrlesssql13.jpg "Entire Flow")

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/svrlesssql14.jpg "Entire Flow")