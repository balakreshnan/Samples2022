# Automated Machine learning Vision using UI

## Azure Machine learning UI for Vision based automated machine learning

## pre-requisites

- Azure Storage
- Azure Machine Learning Workspace
- Coco Dataset
- Data labelling project with labels

## Steps

- First create a data label project
- Then draw bounding boxes for few images and tag them with labels

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision1.jpg "Architecture")

- Above project was created using open source imaeg dataset
- Once you have labels you can label images
- Once labeled let the automatic labelling run
- Now based on what automated labeller has labelled, confirm them
- Now Export the dataset with Azure ML dataset

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision2.jpg "Architecture")

## Automated Machine learning Vision using UI

- Now go to Automated ML in the left Navigation pane
- Create a new automated ML project
- Select the coco dataset that we exported from data labelling project

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision3.jpg "Architecture")

- Then click Next
- Create a new experiment as below image
- Select the lable list column for Target label
- Select compute cluster for compute options
- Only select GPU based compute choice is available
- In my example i am using Instance Segmentation

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision4.jpg "Architecture")

- Next option to Select algorithm

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision5.jpg "Architecture")

- Provide the hyperparameters for the algorithm

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision6.jpg "Architecture")

- Now there is choice to add more algorithms

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision7.jpg "Architecture")

- Provide samplings, Iterations, Early Stopping and concurrent iterations
- Concurrent iteration allows the algorithm to run on multiple compute nodes
- Now click Next
- Provide validation data set if you have
- In my example i didn't have one so i am skipping this step and going to click Finish

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision8.jpg "Architecture")

- Now click Finish
- Above process will create a new job/experiment.
- Now go to Jobs and click on the job you just created
- You should see the job running.

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision9.jpg "Architecture")

- Now wait for 10 minutes for the job to complete

- Once the job complete, check the output

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision10.jpg "Architecture")

- Job output

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision13.jpg "Architecture")

- Here are the metrics for the job

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision14.jpg "Architecture")

- Model outputs

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision15.jpg "Architecture")

- Model output

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision16.jpg "Architecture")

- Compare metrics with different algorithms
- Select Job runs to compare

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision11.jpg "Architecture")

- Click Compare Preview

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/automlvision12.jpg "Architecture")