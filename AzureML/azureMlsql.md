# Azure Machine learning notebook write back to Azure SQL using pyodbc

## Read from Azure SQL as dataset and write back to Azure SQL using pyodbc

## prerequisites

- Azure SQL Databse
- Azure ML Service
- Azure storage account

## First create a Azure SQL database

- Log into Azure SQL
- Go to Query explorer
- Create a database if not exists
- Create a table called imagelist
- Insert dummy data
- Azure cognitive service computer vision

```
create table dbo.imagelist
(
id bigint identity(1,1),
imageurl varchar(500),
imagetoprocess int,
imageupdatedt datetime
)

insert into dbo.imagelist(imageurl,imagetoprocess,imageupdatedt) values 
('https://bingvsdevportalprodgbl.blob.core.windows.net/demo-images/cf73f606-aebe-42b4-b3bf-baba8cc357c1.jpg',0, '2022-02-07 15:49.00')
insert into dbo.imagelist(imageurl,imagetoprocess,imageupdatedt) values 
('https://dedeleads.com/wp-content/uploads/2021/02/data-entry.jpg',0, '2022-02-07 15:49.00')
insert into dbo.imagelist(imageurl,imagetoprocess,imageupdatedt) values 
('https://i.ytimg.com/vi/I8_d54UZOQI/maxresdefault.jpg',0, '2022-02-07 15:49.00')

Select * from dbo.imagelist;

update dbo.imagelist set imagetoprocess = 0
```

## Notebook Code

- Create a dataset with Azure SQL
- Connect to the above SQL
- Use proper sql username and password
- Now go to AML notebook and create a new one
- Use python 3.6 with Azure ML
- install pyodbc libraries

```
!sudo apt-get --assume-yes install python python-pip gcc g++ build-essential
```

- load the dataset

```
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
resource_group = 'RGName'
workspace_name = 'AMLWorkspaceName'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='imagelistest')
dataset.to_pandas_dataframe()
```

- Convert to pandas dataframe

```
df = dataset.to_pandas_dataframe()
```

- now setup Azure Cognitive Services computer vision api

```
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

import os
region = 'centralus'
key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

credentials = CognitiveServicesCredentials(key)
client = ComputerVisionClient(
    endpoint="https://" + region + ".api.cognitive.microsoft.com/",
    #endpoint = "https://aienggtext.cognitiveservices.azure.com/",
    credentials=credentials
)
```

- now create a function to process using Cognitive Services

```
def processimage(url):
    job = client.recognize_printed_text(url,detect_orientation=True, language="en")

    #print(job.regions[0].lines[0].words[0])
    line = 0

    lines = job.regions[0].lines
    #print(lines)
    for line in lines:
        line_text = " ".join([word.text for word in line.words])
        line = 1
        #print(line_text)
    
    return line
```

- Invoke pyodbc

```
import pyodbc
server = 'dbservername.database.windows.net'
database = 'dbname'
username = 'username'
password = 'password'   
driver= '{ODBC Driver 17 for SQL Server}'
```

- process data and update azure sql

```
import pyodbc

# Specifying the ODBC driver, server name, database, etc. directly
#cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=testdb;UID=me;PWD=pass')
with pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password) as conn:
    with conn.cursor() as cursor:
        for index, row in df.iterrows():
            #print (row["id"], row["imageurl"])
            url = row["imageurl"]
            output = processimage(row["imageurl"])
            df["imagetoprocess"] = 1
            #print(output)
            sql = "update dbo.imagelist set imagetoprocess = " + str(1) + "  where imageurl = '" + row["imageurl"] + "'"
            #print(sql)
            cursor.execute(sql)
    conn.commit()
```