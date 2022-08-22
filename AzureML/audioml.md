# Azure Machine learning processing Audio Deep Learning Model - Custom

## Build Deep learning models with audio labelled data set

## Pre Requistie

- Azure Account
- Azure Storage
- Azure Machine learning
- Kaggle data set from - https://www.kaggle.com/datasets/chrisfilo/urbansound8k
- Code i tested was from - https://www.section.io/engineering-education/machine-learning-for-audio-classification/
- Idea here is to show that we can process open source as is using Azure Machine learning
- Above kaggle already has labelled dataset
- If there is no data set we use tools like Audacity to grab the audio and label and export label as text file

## Code

- Create a notebook with python 3.8 and Azure ML
- This has tensorflow installed
- install library

```
pip install librosa
```

- Print AML version

```
import azureml.core
print(azureml.core.VERSION)
```

- when i wrote this code the version was 1.43.0
- print tensorflow version
  
```
import tensorflow as tf
print(tf.__version__)
```

- was 2.2.0
- import

```
import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
%matplotlib inline
import opendatasets as od
```

- download kaggle
- Get the username and token

```
od.download("https://www.kaggle.com/datasets/chrisfilo/urbansound8k/download?datasetVersionNumber=1")
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/audio1.jpg "Architecture")

- Read the audio file

```
file_name='urbansound8k/fold5/100263-2-0-121.wav'

audio_data, sampling_rate = librosa.load(file_name)
#librosa.display.waveplot(audio_data,sr=sampling_rate)
ipd.Audio(file_name)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/audio2.jpg "Architecture")

- display audio content

```
audio_data
```

```
audio_dataset_path='urbansound8k/'
metadata=pd.read_csv('urbansound8k/UrbanSound8K.csv')
metadata.head()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/audio3.jpg "Architecture")

```
metadata['class'].value_counts()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/audio4.jpg "Architecture")

- Feature extraction

```
mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=40)
mfccs
```

- now create a function to extract more features to use in Deep learning

```
def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features
```

- now fill the class with features

```
from tqdm import tqdm
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])
```

- display features

```
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head(10)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/audio5.jpg "Architecture")

- convert to list for us to use in deep learning algorithmns

```
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())
```

- Encoding categorical features

```
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
```

- Split for training and testing

```
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
```

- now get create the network for Neural network

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

num_labels=y.shape[1]

model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))
```

```
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
model.summary()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/audio6.jpg "Architecture")

- Run the model training using fit 

```
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 200
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

#model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/audio7.jpg "Architecture")

- now calculate metrics

```
test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])
```

- Take a test sample
- predict and see the output

```
filename="urbansound8k/fold8/103076-3-0-0.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)
predicted_label=model.predict(mfccs_scaled_features)
print(predicted_label)
classes_x=np.argmax(predicted_label,axis=1)
prediction_class = labelencoder.inverse_transform(classes_x)
prediction_class
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/audio8.jpg "Architecture")