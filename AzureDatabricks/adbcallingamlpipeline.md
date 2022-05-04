# Azure Databrick executing AML batch published pipeline

## Use AML published endpoint using Azure databricks with AML SDK

## Code

- Check AML version

```
import azureml.core
print(azureml.core.VERSION)
```

- Lets configure service principal
- Make sure svc account as contributor role in AML workspace

```
from azureml.core.authentication import ServicePrincipalAuthentication

sp = ServicePrincipalAuthentication(tenant_id="xxxxxx", # tenantID
                                    service_principal_id="xxxxxxxxxxxxxx", # clientId
                                    service_principal_password="xxxxxxxxxxxxx") # clientSecret
```

- Now connect the AML workspace from Azure databricks

```
from azureml.core import Workspace

ws = Workspace.get(name="amlworkspacename",
                   auth=sp,
                   subscription_id="subscriptionID",
                   resource_group="resourcegroupname")
ws.get_details()
```

- Now get the published endpoint and run as experiment
- This is because it's a batch endpoint

```
from azureml.pipeline.core import PipelineEndpoint

pipeline_endpoint_by_name = PipelineEndpoint.get(workspace=ws, name="Titanic1batchinference")
run_id = pipeline_endpoint_by_name.submit("Titanic1batchinference")
print(run_id)
```

- Now check the AML workspace
- Go to Experiment and you should see the experiment and it's run