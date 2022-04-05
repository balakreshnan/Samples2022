# Azure Machine learning pytorch sample

## Use Titanic data set to train a model using pytorch

## Use Case

- Show how to use pytorch to train a model
- Use titanic data set to train a model
- This is only to show how to
- Not intended for production use
- Make sure any data engineering is done for training should also be implemented for batch scoring

## Prerequisites

- Azure Storage
- Azure Machine learning account
- Titanic data set in Data folder in this repo

## Train

- Create a Azure ML Notebook with pytorch and tensorflow with pytorch 3.8

## Code

```
import azureml.core
print(azureml.core.VERSION)
```

- import the libraries

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

- Read the file

```
df = pd.read_csv('titanic3.csv')
```

- Convert data type

```
df.age = df.age.astype(float)
df.fare = df.fare.astype(float)
```

```
df.isnull().sum()
```

- Take only necessary columns

```
data = df[['pclass','survived','sex', 'age', 'fare', 'embarked', 'sibsp']]
```

- calculate mean for age and fair

```
mean_value=data['age'].mean()
data['age'].fillna(value=mean_value, inplace=True)
mean_value=data['fare'].mean()
data['fare'].fillna(value=mean_value, inplace=True)
df2=data.dropna().reset_index(drop=True)
df2
```

```
df2.age = df2.age.astype(float)
df2.fare = df2.fare.astype(float)
sex1={'male':1, 'female':2}
df2.sex=df2.sex.map(sex1)
```

- Split data set as features and labels

```
y = df2["survived"]
features = ["pclass", "sex", "age", "fare", "sibsp"]
X = df2[features]
```

- Now Train, test for model

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, shuffle=True, random_state=1, stratify=y)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
```

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
```

- Model parameters

```
EPOCHS = 45
BATCH_SIZE = 64
LEARNING_RATE = 0.001
```

```
## train data
class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_data = trainData(torch.FloatTensor(X_train.values), 
                       torch.FloatTensor(y_train.values))
## test data    
class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    

test_data = testData(torch.FloatTensor(X_test.values))
```

```
#initialise data loader
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)
```

- Create the neural network

```
class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(num_feature, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x
```

```
num_feature = len(X.columns)
```

```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```

```
model = binaryClassification()
model.to(device)
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

```
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc
```

```
model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        

    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
```

```
y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
```

```
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred_list)
print(accuracy)
```

```
confusion_matrix(y_test, y_pred_list)
print(classification_report(y_test, y_pred_list))
```

```
df3=pd.DataFrame({'Actual': y_test, 'Predicted':y_pred_list})
df3
```

```
from sklearn.decomposition import PCA

pca_val = PCA(n_components=2)
principalComponents_val = pca_val.fit_transform(X_test)
```

```
plt.figure(figsize = (12, 8))
plt.scatter(principalComponents_val[:, 0], principalComponents_val[:,1], c = y_pred == y_test - 1, alpha = .8, s = 50)
```

```
# Specify a path
PATH = "titanic_dict_model.pt"

# Save
torch.save(model, PATH)
```

## Inference

## Code

```
# Specify a path
PATH = "titanic_dict_model.pt"
```

```
# Load
modeleval = torch.load(PATH)
modeleval.eval()
```

```
scoredata = pd.read_csv('titanic3.csv')
```

```
scoredata.age = scoredata.age.astype(float)
scoredata.fare = scoredata.fare.astype(float)
mean_value=scoredata['fare'].mean()
scoredata['fare'].fillna(value=mean_value, inplace=True)
mean_value=scoredata['age'].mean()
scoredata['age'].fillna(value=mean_value, inplace=True)
```

- filter only necessary columns

```
scoredata = scoredata[['pclass','survived','sex', 'age', 'fare', 'embarked', 'sibsp']]
```

```
scoredata2=scoredata.dropna().reset_index(drop=True)
```

```
sex1={'male':1, 'female':2}
scoredata2.sex=scoredata2.sex.map(sex1)
```

```
y = scoredata2["survived"]
features = ["pclass", "sex", "age", "fare", "sibsp"]
X = scoredata2[features]
```

- Predict

```
prediction = modeleval(torch.FloatTensor(X_train.values)).cpu().detach().numpy().flatten()
```

- predicted values

```
y_test_pred = torch.sigmoid(torch.FloatTensor(y.values))
y_pred_tag = torch.round(torch.FloatTensor(y.values))
```

```
df4=pd.DataFrame({'Actual': y, 'Predicted':y_pred_tag})
df4
```

## done