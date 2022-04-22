# Azure Databricks MLFLOW Deploy Notebook Code changes using Azure DevOps

## End to End deploy between environments and run the notebook

## MLOPS

- MLOPS can be split into 3 different parts
- 1) MLOPS infrastructure components. This is only when infrastructure is needed. Doesn't have to run for every code deployment
- 2) Code/Notebook changes deployed to dev, QA and production environment
- 3) Run the Training and Test, and then deploy the model to QA, Production Environment
- Usually Software development only has 2 parts infra and then code deployment
- With Custom Machine learning, we need to have the 3rd part to run the training and deploy the model to be consumed

## Prerequisites

- Azure account
- Azure Databricks Account
- Azure storage
- Azure DevOps
- Github Repo
- Assumtion here is Azure databricks is connected to the Repo and notebook is saved in repo
- Sample Repo - https://github.com/balakreshnan/adbgithubactionfy23
- The Code is all open source and from documentation, there is no sensitive data
- Please substitute your notebook with all security needed

## Azure DevOps

- Lets first connect to the github repo
- Lets Create a New Release Pipeline
- Create a blank tempalte
- Click Add an artifact
- Select Github
- Type the connected repo name
- Branch name
- Select the Repo name
- Select the Latest version

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops1.jpg "Architecture")

- Name the Release ADBNotebookRun
- Next we create 3 stages

### Dev

- Lets add a stage
- Name it as Dev
- Leave the agent as default
- Click Job tasks
- Now we need 4 tasks
- First search for python version and add it to release pipeline

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops2.jpg "Architecture")

- Type the version as 3.7
- Next Search for Configure databricks cli
- Open the data bricks workspace URL and copy the URL with o=######
- Then in Databricks workspace go to Settings -> user settings -> Generate Token -> Copy the token
- Save the token somewhere, other wise we have to recreate new token everytime
- Paste the token and workspace URL in the text box

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops3.jpg "Architecture")

- Assumption here is cluster was created when infra was deployed.
- We are going to start the cluster
- In Databricks workspace go to compute and select the cluster created
- On the right top corner click JSON
- Copy the cluster id

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops4.jpg "Architecture")

- Next Execute cluster from Dev cluser
- Assumption here is since development was done, there is no need to copy again
- select the appropriate path where notebooks are stored

```
/Repos/xxxxx/adbgithubactionfy23/notebooks/Users/xxxx/mlflow/ML End-to-End Example (Azure)
```

- xxx denotes username or email
- Also provide the cluster id for the development environment

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops5.jpg "Architecture")

### QA/Stage

- next add a new stage and name is QA/Stage
- Assumption here is qa environment is already created
- Also cluster should also be created
- Now add task and search for Configure databricks cli
- Make sure you have token created and also the workspace URL http://adp-xxxx.databricks.net/o=xxxxxxxxxxxx format

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops6.jpg "Architecture")

- Start the cluster now
- bring Starting cluster task
- Provide the Cluster ID

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops7.jpg "Architecture")

- Next Bring deploy notebook to workspace
- We need to deploy the notebook from github artifact to workspace
- Select the notebook folder from artifact we created
- Provide the Workspace Folder

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops8.jpg "Architecture")

- Next execute the notebook to make sure it's deployed and also we can execute and deploy model
- Notebook will run through end to end, training, testing and store the experiment details in mlflow and deploy the model
- Note the Notebook URL start with /Repos/xxxxx/adbgithubactionfy23/notebooks/Users/xxxx/mlflow/ML End-to-End Example (Azure)
- This ensures we are using the github repo code.

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops9.jpg "Architecture")

### Production

- next add a new stage and name is Production
- Assumption here is production environment is already created
- Also cluster should also be created
- Now add task and search for Configure databricks cli
- Make sure you have token created and also the workspace URL http://adp-xxxx.databricks.net/o=xxxxxxxxxxxx format

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops10.jpg "Architecture")

- Start the cluster now
- bring Starting cluster task
- Provide the Cluster ID

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops11.jpg "Architecture")

- Next Bring deploy notebook to workspace
- We need to deploy the notebook from github artifact to workspace
- Select the notebook folder from artifact we created
- Provide the Workspace Folder

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops12.jpg "Architecture")

- Next execute the notebook to make sure it's deployed and also we can execute and deploy model
- Notebook will run through end to end, training, testing and store the experiment details in mlflow and deploy the model
- Note the Notebook URL start with /Repos/xxxxx/adbgithubactionfy23/notebooks/Users/xxxx/mlflow/ML End-to-End Example (Azure)
- This ensures we are using the github repo code.

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops13.jpg "Architecture")

- Save the release
- Click Create Release
- Wait for the release to run
- Here is the entire release should look like

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops14.jpg "Architecture")

- now wait for the release to complete
- Once completed here is what you see

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops15.jpg "Architecture")

- Click on the stage logs to see the progress

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureDatabricks/images/adbdevops16.jpg "Architecture")