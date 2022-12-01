# Move Microsoft Graph metadata to Azure Data Explorer

## Pre-requisites

- Azure Account
- Azure Data Explorer Cluster
- Azure Service principal
- Create a secret
- Assign API permission for graph to user.read in delegate and application permissions
- Scope is only to move the meta data from microsoft graph to ADX
- Azure Machine Learning Workspace
- Create a compute cluster

## Code

- install libraries

```
pip install azure-kusto-ingest
```

- Next imports

```
from azure.identity import InteractiveBrowserCredential
from msgraph.core import GraphClient
```

```
import json
from configparser import SectionProxy
from azure.identity import DeviceCodeCredential, ClientSecretCredential
from msgraph.core import GraphClient
```

```
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoServiceError
from azure.kusto.data.helpers import dataframe_from_result_table
from azure.kusto.data.exceptions import KustoServiceError
from azure.kusto.data.helpers import dataframe_from_result_table
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder,ClientRequestProperties
from azure.kusto.ingest import QueuedIngestClient, IngestionProperties, FileDescriptor, BlobDescriptor, ReportLevel, ReportMethod
```

- now set the scope

```
graph_scopes = "User.Read"
```

- set client id, tenant id and secret

```
tenant_id = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
client_id = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
client_secret = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

- now setup for authentication

```
client_credential = ClientSecretCredential(tenant_id, client_id, client_secret)
```

- now setup the graph client

```
user_client = GraphClient(credential=client_credential,
                                      scopes=['https://graph.microsoft.com/.default'])
```

- now get all user data

```
result = user_client.get('/users')
print(result.json())
```

- to display json output

```
result1 = result.json()
for i in result1['value']:
    #print(i)
    print(i["givenName"], i["mail"], i["mobilePhone"], i["officeLocation"], i["userPrincipalName"], i["id"])
    print('\n')
```

- let convert to dataframe

```
import pandas as pd

#df = pd.read_json(result1)
df_nested_list = pd.json_normalize(result1, record_path =['value'])
```

- now configure adx information

```
AAD_TENANT_ID = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
KUSTO_URI = "https://adxname.region.kusto.windows.net"
KUSTO_INGEST_URI = "https://ingest-adxname.region.kusto.windows.net"
KUSTO_DATABASE = "Benchmark"
```

- Invoke the Kusto client

```
kcsb_ingest = KustoConnectionStringBuilder.with_aad_application_key_authentication(
                KUSTO_INGEST_URI, client_id, client_secret, tenant_id)  
```

```
KUSTO_INGESTION_CLIENT = QueuedIngestClient(kcsb_ingest)
```

- csv mapping

```
csvmapping = '[  {"Column": "businessPhones", "Properties": {"Ordinal": "0"}},  {"Column": "displayName", "Properties": {"Ordinal": "1"}},  {"Column": "givenName", "Properties": {"Ordinal": "2"}},  {"Column": "jobTitle", "Properties": {"Ordinal": "3"}},  {"Column": "mail", "Properties": {"Ordinal": "4"}},  {"Column": "mobilePhone", "Properties": {"Ordinal": "5"}},  {"Column": "officeLocation", "Properties": {"Ordinal": "6"}},  {"Column": "preferredLanguage", "Properties": {"Ordinal": "7"}},  {"Column": "surname", "Properties": {"Ordinal": "8"}},  {"Column": "userPrincipalName", "Properties": {"Ordinal": "9"}},  {"Column": "id", "Properties": {"Ordinal": "10"}}]'
```

- Setup ingestion properties

```
from azure.kusto.data import KustoConnectionStringBuilder, DataFormat
```

```
ingestion = IngestionProperties(database="Benchmark", table="graphdata", data_format=DataFormat.CSV, ingestion_mapping_kind=None)
```

- now ingest the data

```
QueuedIngestClient.ingest_from_dataframe(KUSTO_INGESTION_CLIENT,df_nested_list, ingestion)
```

- if you want to write the data to csv file

```
df_nested_list.to_csv('graphdata.csv', header=True)
```