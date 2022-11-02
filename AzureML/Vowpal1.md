# Vowpal Wabbit on Azure ML

## Example to run Vowpal Wabbit on Azure ML

## pre-requisites

- Azure account
- Azure machine learning Service

## Steps

- Create a notebook with python 3.8 with Azure ML as kernel
- Install vowpal wabbit

```
pip install vowpalwabbit
```

- Load the necessary libraries

```
import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.datasets
import vowpalwabbit
```

- now load the iris data set into dataframe

```
iris_dataset = sklearn.datasets.load_iris()
iris_dataframe = pd.DataFrame(
    data=iris_dataset.data, columns=iris_dataset.feature_names
)
# vw expects labels starting from 1
iris_dataframe["y"] = iris_dataset.target + 1
training_data, testing_data = sklearn.model_selection.train_test_split(
    iris_dataframe, test_size=0.2
)
```

- Format feature function

```
def to_vw_format(row):
    res = f"{int(row.y)} |"
    for idx, value in row.drop(["y"]).iteritems():
        feature_name = idx.replace(" ", "_").replace("(", "").replace(")", "")
        res += f" {feature_name}:{value}"
    return res
```

- diplay few rows

```
for ex in training_data.head(10).apply(to_vw_format, axis=1):
    print(ex)
```

- Create vowpall worksspace
- Run training with multiple samples

```
vw = vowpalwabbit.Workspace("--oaa 3 --quiet")

# learn from training set with multiple passes
for example in training_data.apply(to_vw_format, axis=1):
    vw.learn(example)

# predict from the testing set
predictions = []
for example in testing_data.apply(to_vw_format, axis=1):
    predicted_class = vw.predict(example)
    predictions.append(predicted_class)
```

- now Predict the output

```
accuracy = len(testing_data[testing_data.y == predictions]) / len(testing_data)

print(f"Model accuracy {accuracy}")
```

- output should be 

```
Model accuracy 0.7333333333333333
```