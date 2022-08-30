# Responsible AI with Titanic dataset using Azure Machine Learning

## Using Azure Machine Learning to show case Responsible AI with Titanic dataset

## Pre-requistie

- Azure Account
- Azure Storage
- Azure machine learning Service
- titanic.csv file - Data/titanic.csv

## Code

- Check the AML version

```
# Check core SDK version number
import azureml.core

print("SDK version:", azureml.core.VERSION)
```

- Enabling diagnostics

```
from azureml.telemetry import set_diagnostics_collection

set_diagnostics_collection(send_diagnostics=True)
```

- Load workspace config

```
import azureml.core
from azureml.core import Workspace
import pandas as pd

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))
```

- Now load the dataset

```
df= pd.read_csv('./Data/titanic.csv')
print(df.shape)
print(df.columns)
```

## Feature Engineering

- Find duplicates

```
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
df.isnull().sum()
```

- Show data types inside data frame

```
df.dtypes
```

- Create a new column for categorical value

```
df['Loc']= df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'X')
```

- Now drop columns

```
df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
```

- Display the dataframe to validate the columns and data

```
df.head()
```

- Find null count

```
df.isnull().sum()
```

- Create a new column 

```
df.loc[:,'GroupSize'] = 1 + df['SibSp'] + df['Parch']
```

- Fill null column value with 'S'

```
df['Embarked'] = df['Embarked'].fillna('S')
```

- Convert Sex into numeric values

```
df['Sex'] = df['Sex'].apply({'male':1, 'female':2}.get)
```

- For embarked also please change to numeric

```
df['Embarked'] = df['Embarked'].apply({'C':1, 'Q':2, 'S':3}.get)
```

- Change for location category

```
df['Loc'] = df['Loc'].apply({'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'X':9}.get)
```

- Display the data and validate

```
df.head()
```

- Set the label column
- Drop unnecessary columns

```
LABEL = 'Survived'
columns_to_keep = ['Pclass', 'Sex','Age', 'Fare', 'Embared', 'Deck', 'GroupSize']
columns_to_drop = ['Name','SibSp', 'Parch', 'Survived']
df_train = df
df = df_train.drop(['Name','SibSp', 'Parch', 'PassengerId'], axis=1)

df.head(5)
```

- if you want to convert all categorical string columns to numeric columns

```
df = pd.get_dummies(df)
```

## Model Training

- Create a folder

```
import os
script_folder = os.path.join(os.getcwd(), "train_remote")
print(script_folder)
os.makedirs(script_folder, exist_ok=True)
```

- now split the feature and label columns

```
y = df['Survived']
df = df.drop(['Survived'], axis=1)
```

- set the target column name (label)

```
target_feature = 'Survived'
```

- now split the data for training and testing

```
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=7)

train_data = X_train.copy()
test_data = X_test.copy()
train_data[target_feature] = y_train
test_data[target_feature] = y_test
```

- now import the model to use

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
```

- Since it's a classification problem, we will use Random Forest Classifier

```
model = RandomForestClassifier(max_depth=2, random_state=0)
model.fit(X_train, y_train)
```

- now predict the test data

```
pred = model.predict(X_test)
pred_proba = model.predict_proba(X_test)
```

- Now evaluate the model

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

```
# View accuracy score
accuracy_score(y_test, pred)
```

- Confusion Matrix

```
# View confusion matrix for test data and predictions
confusion_matrix(y_test, pred)
```

- Classification Report

```
# View the classification report for test data and predictions
print(classification_report(y_test, pred))
```

## Responsible AI

- now import rais module

```
from raiwidgets import ResponsibleAIDashboard
from responsibleai import RAIInsights
```

- Configure RAI insights
- Larger the data set - takes a long time

```
rai_insights = RAIInsights(model, train_data[:20], test_data[:20], target_feature, 'classification',
                               categorical_features=[])
```

- i am only taking 20 rows
- Now let's add what to analyze

```
# Interpretability
rai_insights.explainer.add()
# Error Analysis
rai_insights.error_analysis.add()
# Counterfactuals: accepts total number of counterfactuals to generate, the range that their label should fall under, 
# and a list of strings of categorical feature names
```

- Calculate RAI insights

```
rai_insights.compute()
```

- Create the Responsible AI Dashboard

```
ResponsibleAIDashboard(rai_insights)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titanicrai5.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titanicrai7.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titanicrai8.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titanicrai9.jpg "Architecture")

## Data Profiling

- Using pandas profiling tool

```
from pandas_profiling import ProfileReport
```

- Set the report

```
profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
```

- Display the report

```
profile.to_widgets()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titanicrai1.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titanicrai2.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titanicrai3.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titanicrai4.jpg "Architecture")


## Conclusion

- End to end Training and validation process for data science project.