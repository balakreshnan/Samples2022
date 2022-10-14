# Azure Machine learning Federated learning

## Federated learning sampling introduction

## Prerequisites

- Azure Account
- Azure Machine Learning Service
- clone the github repo - https://github.com/balakreshnan/azure-ml-federated-learning
- Go to quick start - https://github.com/balakreshnan/azure-ml-federated-learning/blob/main/docs/quickstart.md

## Federated learning Steps

- Create a workspace
- Create a compute instance
- or local workstation is also fine
- there is bicep code to create the workspace and compute instance

## Code

- Login into Azure CLI

```
az login --tenant <tenant-id>
az account set --name <subscription name>
```

- create a reource group

```
az group create --name fltest --location eastus
```

- now create the resource
- Change the name fldemo to something else

```
az deployment group create --template-file ./mlops/bicep/open_sandbox_setup.bicep --resource-group fltest --parameters demoBaseName="fldemo22"
```

- wait until the deployment is complete
- now run the python code to setup the environment

```
python -m pip install -r ./examples/pipelines/fl_cross_silo_literal/requirements.txt
```

- now update the config.json

```
{
    "subscription_id": "<subscription-id>",
    "resource_group": "<resource-group>",
    "workspace_name": "<workspace-name>"
}
```

- now run the training code

```
python ./examples/pipelines/fl_cross_silo_literal/submit.py --example MNIST --submit
```

- wait for the training to complete

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/fltest1.jpg "Architecture")

- Metric view

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/fltest2.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/fltest3.jpg "Architecture")

- Steps review

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/fltest4.jpg "Architecture")