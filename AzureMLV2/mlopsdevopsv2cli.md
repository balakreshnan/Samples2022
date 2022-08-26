# Azure Machine learning CLI V2 deployment using Azure DevOps

## Azure DevOps CLI V2 deployment using Azure DevOps

## pre-requistie

- Azure Account
- Azure Storage
- Azure Machine learning workspace
- Azure DevOps
- Azure CLI V2 with ml extentions
- Public template used - https://github.com/MicrosoftLearning/mslearn-mlops

## Training and Deployment Process

- Log into Azure DevOps
- Create a new project
- Create new pipeline
- Select Github as source
  
![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/devopscli1.jpg "Architecture")

- Click Next
- Select the repository

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/devopscli2.jpg "Architecture")

- Next Select the Start pipeline

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/devopscli3.jpg "Architecture")

- Click Next
- Now you will see the pipelie yaml file
- Now Click Variable in the pipeline
- you need to have Service principal clientid, secret and tenantid
- Give contributor permission to AML workspace to deploy code and managed endpoint

## Training

- Code breakdown
- First create a variable

```
variables:
- name: runid
  value: ""
```

- First Set python version to 3.8

```
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.8'
    addToPath: true
    architecture: 'x64'
```

- Install az ml extension

```
- task: Bash@3
  inputs:
    targetType: 'inline'
    script: |
      # Write your commands here      
      az extension add -n ml -y
```

- Next is Azure login using service principal to automate deployment
- Necessary logins are passed as variable and not hardcoded

```
- task: Bash@3
  inputs:
    targetType: 'inline'
    script: |
      # Write your commands here      
      az login --service-principal -u $(clientid) -p $(secret) --tenant $(tenantid)
```

- Next is show az version

```
- task: Bash@3
  inputs:
    targetType: 'inline'
    script: |
      # Write your commands here
      az --version
```

- Next is running training code

```
- task: Bash@3
  inputs:
    targetType: 'inline'
    script: |
      cd src
      run_id=$(az ml job create --file job.yml --resource-group "rg-dev-mlops" --workspace-name "mlw-dev-mlops" --query name -o tsv)
      echo "##vso[task.setvariable variable=runid;]$run_id"
      echo "Run ID from above Job Run is $run_id stored in environment variable"
```

- Get the Run id and store in variable for future use in the deployment if needed as above
- Display the Run Id

```
- task: Bash@3
  inputs:
    targetType: 'inline'
    script: 'echo "Run ID from above Job Run is $run_id stored in environment variable"'
```

- Register the model

```
- task: bash@3
  inputs:
    targetType: 'inline'
    script: |
      cd src
      az ml model create -n diabetes-data-example -p runs:/<runid>/model --type mlflow_model --resource-group rg-dev-mlops --workspace-name mlw-dev-mlops
```

## Deployment

- now delete the endpoint if exists

```
- task: bash@3
  inputs:
    targetType: 'inline'
    script: |
        cd src
        az ml online-endpoint delete --name diabetesendpointbb22 --resource-group rg-dev-mlops --workspace-name mlw-dev-mlops --yes
```

- Create a endpoint

```
- task: bash@3
  inputs:
    targetType: 'inline'
    script: |
        cd src
        az ml online-endpoint create --name diabetesendpointbb22 -f endpoint.yml --resource-group rg-dev-mlops --workspace-name mlw-dev-mlops
```

- Update deployment with traffic

```
- task: bash@3
  inputs:
    targetType: 'inline'
    script: |
        cd src
        az ml online-deployment create --name blue --endpoint diabetesendpointbb22 --file blue-deployment.yml --all-traffic --resource-group rg-dev-mlops --workspace-name mlw-dev-mlops
```

- Now show the details of Managed endpoint

```
- task: bash@3
  inputs:
    targetType: 'inline'
    script: |
        cd src
        az ml online-endpoint show -n diabetesendpointbb22 --resource-group rg-dev-mlops --workspace-name mlw-dev-mlops
```

- Now score the model with sample data

```
- task: bash@3
  inputs:
    targetType: 'inline'
    script: |
        cd src
        az ml online-endpoint invoke --name diabetesendpointbb22 --request-file sample-request.json --resource-group rg-dev-mlops --workspace-name mlw-dev-mlops
```

- if you don't want to use the endpoint, delete the endpoint

## Entire Code block for yml file

- Here is the entire yaml code

```
# Starter pipeline
# Start with a minimal pipeline that you can customize to build and deploy your code.
# Add steps that build, run tests, deploy, and more:
# https://aka.ms/yaml

trigger:
- main

pool:
  vmImage: ubuntu-latest

variables:
- name: runid
  value: ""

steps:

- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.8'
    addToPath: true
    architecture: 'x64'

- task: Bash@3
  inputs:
    targetType: 'inline'
    script: |
      # Write your commands here      
      az extension add -n ml -y

- task: Bash@3
  inputs:
    targetType: 'inline'
    script: |
      # Write your commands here      
      az login --service-principal -u $(clientid) -p $(secret) --tenant $(tenantid)

- task: Bash@3
  inputs:
    targetType: 'inline'
    script: |
      # Write your commands here
      az --version
      # az account show

- task: Bash@3
  inputs:
    targetType: 'inline'
    script: |
      cd src
      run_id=$(az ml job create --file job.yml --resource-group "rg-dev-mlops" --workspace-name "mlw-dev-mlops" --query name -o tsv)
      echo "##vso[task.setvariable variable=runid;]$run_id"
      echo "Run ID from above Job Run is $run_id stored in environment variable"

- task: Bash@3
  inputs:
    targetType: 'inline'
    script: 'echo "Run ID from above Job Run is $run_id stored in environment variable"'

- task: bash@3
  inputs:
    targetType: 'inline'
    script: |
      cd src
      az ml model create -n diabetes-data-example -p runs:/trainrun1/model --type mlflow_model --resource-group rg-dev-mlops --workspace-name mlw-dev-mlops

- task: bash@3
  inputs:
    targetType: 'inline'
    script: |
        cd src
        az ml online-endpoint delete --name diabetesendpointbb22 --resource-group rg-dev-mlops --workspace-name mlw-dev-mlops --yes

- task: bash@3
  inputs:
    targetType: 'inline'
    script: |
        cd src
        az ml online-endpoint create --name diabetesendpointbb22 -f endpoint.yml --resource-group rg-dev-mlops --workspace-name mlw-dev-mlops

- task: bash@3
  inputs:
    targetType: 'inline'
    script: |
        cd src
        az ml online-deployment create --name blue --endpoint diabetesendpointbb22 --file blue-deployment.yml --all-traffic --resource-group rg-dev-mlops --workspace-name mlw-dev-mlops

- task: bash@3
  inputs:
    targetType: 'inline'
    script: |
        cd src
        az ml online-endpoint show -n diabetesendpointbb22 --resource-group rg-dev-mlops --workspace-name mlw-dev-mlops

- task: bash@3
  inputs:
    targetType: 'inline'
    script: |
        cd src
        az ml online-endpoint invoke --name diabetesendpointbb22 --request-file sample-request.json --resource-group rg-dev-mlops --workspace-name mlw-dev-mlops
```

- Save and Run
- Wait for pipeline run to complete

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/devopscli4.jpg "Architecture")

- here is the job details
- Click on the job run and you will job summary page

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/devopscli6.jpg "Architecture")

- as you can see every task is completed successfully

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/devopscli5.jpg "Architecture")

- Thank you and have fun!