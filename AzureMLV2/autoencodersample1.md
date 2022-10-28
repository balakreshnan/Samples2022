# Azure Machine Learning Auto Encoder Anamoly Detection Sample

## Anamoly Detection using Auto Encoder

## Prerequisites

- Azure Account
- Azure Machine Learning Service

```
Note: This sample is from Tensorflow to show how it works in Azure Machine Learning.
I have not installed any libraries in the Azure Machine Learning environment.
```

- Here is the actual code from
- https://www.tensorflow.org/tutorials/generative/autoencoder
- Also using open source dataset

## Code

- First download the dataset

```
# Download the dataset
dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
dataframe.head()
```

- Split the data into train and test

```
# The last element contains the labels
labels = raw_data[:, -1]

# The other data points are the electrocadriogram data
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)
```

- Process above dataset

```
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)
```

- Set the label

```
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]
```

- Plot the data

```
plt.grid()
plt.plot(np.arange(140), normal_train_data[0])
plt.title("A Normal ECG")
plt.show()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/anamoly1.jpg "Architecture")

```
plt.grid()
plt.plot(np.arange(140), anomalous_train_data[0])
plt.title("An Anomalous ECG")
plt.show()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/anamoly2.jpg "Architecture")

- Create the model

```
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(140, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()
```

- Set the optimizer

```
autoencoder.compile(optimizer='adam', loss='mae')
```

- Train the model

```
history = autoencoder.fit(normal_train_data, normal_train_data, 
          epochs=20, 
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)
```

- plot the loss metrics

```
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/anamoly3.jpg "Architecture")

- now the error

```
encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(normal_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(140), decoded_data[0], normal_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/anamoly4.jpg "Architecture")

```
encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(anomalous_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(140), decoded_data[0], anomalous_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/anamoly5.jpg "Architecture")

- now predict

reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

```
plt.hist(train_loss[None,:], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/anamoly6.jpg "Architecture")

- Threshold calcualtions

```
threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

reconstructions = autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

plt.hist(test_loss[None, :], bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/anamoly7.jpg "Architecture")

- Predict function

```
def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))
```

- Predict and print the stats

```
preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureMLV2/images/anamoly8.jpg "Architecture")