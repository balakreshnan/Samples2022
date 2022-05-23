# Responsible AI Implementation on Titanic DataSet

## Apply responsible AI to Titanic DataSet

## Prerequisites

- Azure account
- Azure Storage account
- Azure Machine Learning Service
- Titanic dataset

## Code

- Include RAI imports

```
from raiwidgets import ResponsibleAIDashboard
from responsibleai import RAIInsights
```

- Scikit includes

```
import shap
import sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
```

- Now configure AML workspace

```
import azureml.core
from azureml.core import Workspace
import pandas as pd

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))
```

- Load the data

```
df= pd.read_csv('./Data/titanic.csv')
print(df.shape)
print(df.columns)
```

## Feature Engineering

- Find nulls

```
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
df.isnull().sum()
```

- Apply missing Cabin as X

```
df['Loc']= df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'X')
```

- Drop few columns inplace

```
df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
```

- Create a new column by combining 2 columns

```
df.loc[:,'GroupSize'] = 1 + df['SibSp'] + df['Parch']
```

- Fill empty with vaues

```
df['Embarked'] = df['Embarked'].fillna('S')
```

- Split features and labels

```
LABEL = 'Survived'
columns_to_keep = ['Pclass', 'Sex','Age', 'Fare', 'Embared', 'Deck', 'GroupSize']
columns_to_drop = ['Name','SibSp', 'Parch', 'Survived']
df_train = df
df = df_train.drop(['Name','SibSp', 'Parch', 'PassengerId'], axis=1)

df.head(5)
```

- Set training folders

```
import os
script_folder = os.path.join(os.getcwd(), "train_remote")
print(script_folder)
os.makedirs(script_folder, exist_ok=True)
```

- Save the above feature engineered data set

```
df.to_csv('./train_remote/titanic.csv')
df.head(2)
```

## Training

- Lets build the data set
- Preporcessing function code

```
def buildpreprocessorpipeline(X_raw):
    categorical_features = X_raw.select_dtypes(include=['object']).columns
    numeric_features = X_raw.select_dtypes(include=['float','int64']).columns

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value="missing")),
                                              ('onehotencoder', OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore'))])
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ], remainder="drop")
    
    return preprocessor
```

- includes for training 

```
import joblib
import pandas as pd
import numpy as np

from azureml.core import Run, Dataset, Workspace, Experiment

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score,roc_curve
from interpret.ext.blackbox import TabularExplainer
from azureml.interpret import ExplanationClient

from raiwidgets import ResponsibleAIDashboard
from responsibleai import RAIInsights

# Calculate model performance metrics
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration
```

- Specified the label

```
LABEL = 'Survived'
y_raw = df[LABEL]
X_raw = df.drop([LABEL], axis=1)
```

- Split the data set

```
 # Train test split
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.3, random_state=0)
```

- now run the model

```
lg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
preprocessor = buildpreprocessorpipeline(X_train)

#estimator instance
clf = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', lg)])

model = clf.fit(X_train, y_train)
```

- Calculate AUC

```
# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
```

- Test accuracy

```
 # calculate test accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
#run.log('Accuracy', np.float(acc))
```

```
%matplotlib inline
```

```
# plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
fig = plt.figure(figsize=(6, 4))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
#run.log_image(name = "ROC", plot = fig)
plt.show()
```

- Now setup columns for RAI toolkit

```
columns = ['Survived']
#df = df.drop('Survived', axis=1, inplace=True)
#df = df.drop(columns, axis=1)
#print(df.columns)
dfcolumns = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Loc', 'GroupSize']
print(dfcolumns)
#client = ExplanationClient.from_run(run)
# Explain predictions on your local machine
tabular_explainer = TabularExplainer(model, X_train, features=dfcolumns)
```

- Now invoke the global explainer

```
global_explanation = tabular_explainer.explain_global(X_test)
```

- Now set the variable for sensitive columns

```
comment = 'Global explanation on regression model trained on boston dataset'
#client.upload_model_explanation(global_explanation, comment=comment, model_id=model)
categorical_features = X_raw.select_dtypes(include=['object']).columns
target_feature = LABEL
train_data = X_train.copy()
test_data = X_test.copy()
train_data[target_feature] = y_train
test_data[target_feature] = y_test
```

- Invoke RAI insights

```
rai_insights = RAIInsights(model, train_data, test_data, LABEL, 'classification', 
                               categorical_features=['Sex','Embarked','Loc'])
```

- Add the analysis

```
# Interpretability
rai_insights.explainer.add()
# Error Analysis
rai_insights.error_analysis.add()
# Counterfactuals: accepts total number of counterfactuals to generate, the range that their label should fall under, 
# and a list of strings of categorical feature names
rai_insights.counterfactual.add(total_CFs=20, desired_class='opposite')
rai_insights.causal.add(treatment_features=['Sex', 'Embarked'])
```

- run the RAI compute
- Compute takes time to process
- There will be a UI limitation to only consume 5000 rows to display

```
rai_insights.compute()
```

- invoke the RAI insights

```
ResponsibleAIDashboard(rai_insights)
```

## RAI Dashboard Analysis

- Error Analysis
- Error Analysis with Error Rate

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai1.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai2.jpg "Architecture")

- Heat Map option

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai3.jpg "Architecture")

- Error Analysis with Accuracy score

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai4.jpg "Architecture")

- Error Analysis with Precision Score

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai5.jpg "Architecture")

- Model Overview

- with Column Sex

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai6.jpg "Architecture")

- With column Age

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai7.jpg "Architecture")

- Data Explorer

- WIth PClass

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai8.jpg "Architecture")

- With Sex

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai9.jpg "Architecture")

- With Age

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai10.jpg "Architecture")

- Feature Importance
- You can analyze predicted outcome and also dataset based on predicted outcome

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai11.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai12.jpg "Architecture")

- Counterfactual Analysis

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai13.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai14.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai15.jpg "Architecture")

- Causal Analysis

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai16.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai17.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/titrai18.jpg "Architecture")