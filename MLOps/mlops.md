# Data Science and ML Ops process

## MLOps in 2022

## Machine learning and data science Ops process

- As Data science field grows and new innovation is expanding the process
- New process to explain and see what the model does
- Check for bias and fariness in the model
- Monitor and maintain data drift and model performance drift
- New tools and automation
- Privacy, governance and security
- This is just a new current version as of 2022 and subjective to change with technology changes

## MLOPS process

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/MLOps/images/ML-MLOPS-Process-2022-2.jpg "Architecture")

## Process explained

- Data science is been successful when we take use case based approach
- Every use case is going let us know what data is needed and from which data sources
- Since the data is sourced from various data source, it's good to build a data lake to land the data
- Once the data is landed in data lake, It's time to do data engineering or feature engineering
- feature engineering is way to organize features needed for modelling
- features can be sources from multiple sources
- feature might be created based on use case
- features can be aggregated based on use case
- Once features are collected and processed next modelling
- Modelling is a process to split data into two parts
- Mostly 70, 30 or 80, 20 precentage split
- Once data is split run through alogrithm
- This is an iterative process either features can be changed or algorithm
- depending on what type of machine learning like (classification, regression or timeseries) various algorithm can be applied
- Validate the model with new dataset and analyze the performance
- Here we will will also deep dive into the model and explain what it's doing
    - Explainability
    - Fairness
    - Error Analysis
    - Counterfactuals
    - Causal Analysis
    - Data Explorer
    - Privacy
- Apply responsible AI to make sure the model is fair and provides best outcomes
- Model that provides best accuracy is choosen to move forward.
- Make sure the model metric and accuracy are stored
- in some cases RUC chart or other visuals are also preserved for further analysis
- When we are ready for deployment, we need to evaulate if the new model is better then previous one for model already created
- If new model or existing model with feature count changes, then we can skip the evaulate check
- Once it passes then Register the model
- When registering model we also register feature store for others to use
- Model metrics like accuracy and other metrics are stored to track model performance
- Optionally we can register the model as market place for other's in the organization to consume
- Next would be Responsible AI
- This section is new
- Here we will will also deep dive into the model and explain what it's doing
    - Explainability
    - Fairness
    - Error Analysis
    - Counterfactuals
    - Causal Analysis
    - Data Explorer
    - Privacy
- All of the above are capability to calculated and saved for the model.
- In development phase this can be applied to each model to figure out which model provides best outcomes
- Then deploy your model as realtime or batch endpoints
- Also save the metric of model inferencing
- Optional features like data drift and model drift can be watched to retrain the model if performance degrades
- Entire process is built with security in mind
- Governance is also built with monitoring to manage the process
- Optional lineage and cataloging is also applied for both training and inferencing