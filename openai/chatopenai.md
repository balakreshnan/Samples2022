# Open AI Chat with new GPT-3.5

## Pre-requisites

- Open AI Account
- http://chat.openai.com
- Get a developer key

## What is GPT 3.5?

```
Models referred to as "GPT 3.5"
GPT-3.5 series is a series of models that was trained on a blend of text and code from before Q4 2021. The following models are in the GPT-3.5 series:

code-davinci-002 is a base model, so good for pure code-completion tasks
text-davinci-002 is an InstructGPT model based on code-davinci-002
text-davinci-003 is an improvement on text-davinci-002
```

- Information available at https://beta.openai.com/docs/model-index-for-researchers
- Look at the limitation and details about chat in this page - https://openai.com/blog/chatgpt/
- Here we are going to look into writing ML code using GPT-3.5

## Create code by asking question

- Create a classification model for titanic dataset with Azure Machine learning Automated ML

```
# Import the necessary libraries
from azureml.core import Workspace, Dataset
from azureml.train.automl import AutoMLConfig

# Load the titanic data into a Dataset object
dataset = Dataset.get_by_name(workspace=workspace, name="titanic")

# Split the data into training and testing sets
train, test = dataset.random_split(percentage=0.8, seed=1)

# Define the AutoML configuration
automl_config = AutoMLConfig(task="classification",
                             primary_metric="AUC_weighted",
                             max_time_sec=3600,
                             n_cross_validations=5,
                             debug_log="automl_errors.log",
                             verbosity=logging.INFO,
                             X=train.drop_columns(columns=["survived"]),
                             y=train["survived"],
                             X_valid=test.drop_columns(columns=["survived"]),
                             y_valid=test["survived"])

# Train the model using the AutoML configuration
automl_run = AutoMLRun(workspace=workspace, config=automl_config)
automl_run.submit(show_output=True)
automl_run.wait_for_completion(show_output=True)

# Get the best model from the AutoML run
best_model = automl_run.get_output()

# Evaluate the performance of the best model
perf = best_model.predict(test.drop_columns(columns=["survived"]))
print(perf.auc())
```

- Now next to try Deep learning models
- So here is my question

```
# Import the necessary libraries
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Load the titanic data into a Pandas DataFrame
df = pd.read_csv("titanic.csv")

# Split the data into training and testing sets
train, test = df.split_frame(ratios=[0.8])

# Define the model architecture
model = Sequential()
model.add(Dense(32, input_dim=6, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(x=train.drop("survived", axis=1), y=train["survived"], epochs=10, batch_size=32)

# Evaluate the performance of the model
_, acc = model.evaluate(x=test.drop("survived", axis=1), y=test["survived"])
print("Test accuracy:", acc)
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/openai/images/openai1.jpg "Architecture")

- Here is another open source model called yolov5
- question - Can you create me a yolov5 deep learning code?

```
# Import the necessary libraries
import torch
from torch import nn
import torchvision
from torchvision import models

# Define the model architecture
model = models.yolo5(num_classes=80)

# Set the model to train mode
model.train()

# Define the loss function
criterion = nn.BCEWithLogitsLoss()

# Use Adam as the optimizer
optimizer = torch.optim.Adam(model.parameters())

# Train the model
for i in range(num_epochs):
    for images, labels in train_dataloader:
        # Forward pass
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Set the model to evaluation mode
model.eval()

# Evaluate the performance of the model
for images, labels in test_dataloader:
    # Forward pass
    outputs = model(images)

    # Calculate the loss
    loss = criterion(outputs, labels)

    # Print the loss
    print("Loss:", loss.item())
```

- This is very interesting and I am sure this will be very useful for developers and data scientists to create code by asking questions