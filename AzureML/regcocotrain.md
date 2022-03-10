# Register coco training dataset to AzureML

## Create a dataset for AML data labelling

## requirements

- Azure account
- Azure Storage account
- Azure Machine Learning service
- Coco dataset URL - http://images.cocodataset.org/zips/train2017.zip

## Steps

- Initiate the AML workspace

```
from azureml.core import Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath

ws = Workspace.from_config()
datastore = Datastore.get(ws, 'workspaceblobstore')
```

- Now code to download the dataset from coco URL

```
import urllib
import zipfile

url = "http://images.cocodataset.org/zips/train2017.zip"
extract_dir = "cocotrain2017"

```

- UnZip the files to local close to 180K images

```
zip_path, _ = urllib.request.urlretrieve(url)
with zipfile.ZipFile(zip_path, "r") as f:
    f.extractall(extract_dir)
```

- Upload to default AML storage

```
 from azureml.core import Dataset, Datastore
 datastore = Datastore.get(ws, 'workspaceblobstore')
 Dataset.File.upload_directory(src_dir='./cocotrain2017', target=(datastore,'cocotrain2017'))
 ```

- Register the data set

```
from azureml.core import Workspace, Datastore, Dataset

# create a FileDataset pointing to files in 'animals' folder and its subfolders recursively
datastore_paths = [(datastore, 'cocotrain2017')]
cocotrain2017_ds = Dataset.File.from_files(path=datastore_paths)

cocotrain2017_ds = cocotrain2017_ds.register(workspace=ws, name='cocotrain2017',  description='Coco training data')
```

- Now go to AML dataset and validate the dataset is created