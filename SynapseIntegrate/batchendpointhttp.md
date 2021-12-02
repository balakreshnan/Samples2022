# Azure Synapse Analytics invoke Azure ML Batch Endpoint using HTTP/Web activity

## How to invoke Azure ML Batch Endpoint using HTTP/Web activity from Synapse Integrate

## Prerequistie

- Azure Account
- Azure Storage
- Azure Machine learning Service
- Create a batch endpoint as per documentation
- This article doesn't show how to deploy batch endpoint.
- https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-batch-endpoint
- Code for batendpoint is available in the above link
- Create Azure synapse analytics workspace
- Create Azure Key vault
- Store Client id, secret and tenant in Azure Keyvault


```
Note: to show how we can do ETL/ELT and then invoke batch endpoint to score ML model.
```

## Architecture / Flow

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/batendpthttp1.jpg "Entire Flow")

## Flow Steps/ACtivities

- Bring web activity

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/batendpthttp2.jpg "Entire Flow")

- Configure the secret URL for Client id stored in Azure Keyvault
- For URL

```
https://keyvaultname.vault.azure.net/secrets/secretname/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx?api-version=7.0
```

- Select GET
- for resource
- For security select managed identity

```
https://vault.azure.net
```

- Create variables

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/batendpthttp3.jpg "Entire Flow")

- Now set the variable

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/batendpthttp4.jpg "Entire Flow")

```
@activity('clientid').output.value
```

- Lets drag web activity again for secret now
- For URL

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/batendpthttp5.jpg "Entire Flow")

```
https://keyvaultname.vault.azure.net/secrets/secretname/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx?api-version=7.0
```

- Select GET
- for resource
- For security select managed identity

```
https://vault.azure.net
```

- Next set the variable

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/batendpthttp6.jpg "Entire Flow")

```
@activity('clientsecret').output.value
```

- now lets get Authorization token

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/batendpthttp7.jpg "Entire Flow")

 - For URL

```
https://login.microsoftonline.com/tenantid/oauth2/token
```

- for body

```
@concat('grant_type=client_credentials&client_id=',variables('clientid'),'&resource=https://management.core.windows.net/&client_secret=',variables('clienttoken'))
```

- now save the token in variable

```
@activity('Getbearertoken').output.access_token
```

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/batendpthttp8.jpg "Entire Flow")

- Now lets get a scoring token for batch endpoint

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/batendpthttp9.jpg "Entire Flow")

```
https://login.microsoftonline.com/tenantid/oauth2/token
```

```
@concat('grant_type=client_credentials&client_id=',variables('clientid'),'&resource=https://ml.azure.com/&client_secret=',variables('clienttoken'))
```

- Save the scoring token in a variable

```
@activity('scoring_token').output.access_token
```

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/batendpthttp10.jpg "Entire Flow")

- now retrieve dataset id

- for URL

```
https://centralus.experiments.azureml.net/dataset/v1.0/subscriptions/<subid>/resourceGroups/<rgname>/providers/Microsoft.MachineLearningServices/workspaces/<workspacename>/saveddatasets/from-data-path
```

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/batendpthttp11.jpg "Entire Flow")

- Add header for authorization

```
@concat('Bearer ',variables('tokenml'))
```

- Body

```
{"DatastoreName":"taxibatch2files","RelativePath":"/example-data/taxibatch.csv"}
```

- now invoke the batch endpoint

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/batendpthttp12.jpg "Entire Flow")

- Now use the Batch endpoint from Azure ML Service
- In Azure ML Go to Endpoint -> Batch endpoint -> select batch endpoint -> click details

```
https://endpointname.centralus.inference.ml.azure.com/jobs
```

- Set headers for Authorization and content-type

```
Authorization: @concat('Bearer ',variables('tokenml'))
```

```
Content-Type: application/json
```

- for Body change to

```
{
  "properties": {
    "dataset": {
      "dataInputType": "DatasetVersion",
      "datasetName": "taxibatch2files",
      "datasetVersion": "1"
    }
  }
}
```

- Commit the changes
- Click debug

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/batendpthttp13.jpg "Entire Flow")

- wait for the flow to complete

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/SynapseIntegrate/Images/batendpthttp14.jpg "Entire Flow")

- Done