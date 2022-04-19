# Azure Machine learning deisgner training and automate batch inference

## Use end to end batch inference using syanpse, azure databricks and AML batch inference pipeline

## Prerequisites

- Azure account
- Azure Machine learning account
- Azure storage account
- Azure databricks account
- Azure synapse workspace account

## Architecture

- Using AML Designer to create a batch inference pipeline
- Automate Batch inferencing

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/amldesigner1.jpg "Architecture")

### Designer Training

- Create a experiment in designer
- Choose computer cluster
- Use open source dataset

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/amldesigner4.jpg "Architecture")

- Click Sumbit and train the model
- Select Create batch inference pipeline
- Create a data store to ADLS gen2 with new dataset with empty file.
- Then add export data
- Save the output as parquet and give a filename

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/amldesigner5.jpg "Architecture")

- after submit and wait for the run to complete
- then click publish

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/amldesigner6.jpg "Architecture")

- Wait for the batch inference endpoint to publish

### End to End automated batch inference

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/amldesigner2.jpg "Architecture")

- Now go to azure synapse analytics
- Now create a pipeline
- Drag Azure databricks and connect to ADB workspace
- Select the notebook - this creates input batch dataset and stores in batchinput container as parquet file
- Then Drag Azure ML and Select the publish pipeline

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/amldesigner7.jpg "Architecture")

- Then drag another Azure databricks and select the notebook to consume batch output and store back in delta table

## Output

- Finalize the batch inference pipeline run

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/amldesigner3.jpg "Architecture")