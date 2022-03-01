# Azure Machine learning designer for K means cluster

## K means Cluster with Designer

## Resources

- Azure Account
- Azure storage account
- Azure Machine Learning workspace
- Create a compute instance
- Data set used is from samples

## Designer

- Overall design flow

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/designer1.jpg "Architecture")

- Drag dataset from dataset section
- Drag select Columns in Dataset
- now bring in Execute python script
- This is to show how python scripting can be done in designer
- Create a covariance matrix

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/designer3.jpg "Architecture")

- create a correlation matrix
- Replace the code below into the box
- Code has both covariance and correlation matrix

```
# imports up here can be used to
import pandas as pd

# The entry point function MUST have two input arguments.
# If the input port is not connected, the corresponding
# dataframe argument will be None.
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1 = None, dataframe2 = None):

    # Execution logic goes here
    print(f'Input pandas.DataFrame #1: {dataframe1}')

    # If a zip file is connected to the third input port,
    # it is unzipped under "./Script Bundle". This directory is added
    # to sys.path. Therefore, if your zip file contains a Python file
    # mymodule.py you can import it using:
    # import mymodule
    import seaborn as sn
    import matplotlib.pyplot as plt

    print(dataframe1.describe())

    corrMatrix = dataframe1.corr()
    print (corrMatrix)
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
    img_file = "corrchart1.png"
    plt.savefig(img_file)

    from azureml.core import Run
    run = Run.get_context(allow_offline=True)
    run.upload_file(f"graphics/{img_file}", img_file)

    covMatrix = dataframe1.cov()
    print (covMatrix)
    sn.heatmap(covMatrix, annot=True)
    plt.show()
    img_file = "covchart1.png"
    plt.savefig(img_file)

    from azureml.core import Run
    run = Run.get_context(allow_offline=True)
    run.upload_file(f"graphics/{img_file}", img_file)

    # Return value must be of a sequence of pandas.DataFrame
    # E.g.
    #   -  Single return value: return dataframe1,
    #   -  Two return values: return dataframe1, dataframe2
    return dataframe1,

```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/designer2.jpg "Architecture")

- Drag the KMeans cluster

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/designer5.jpg "Architecture")

- Bring Train cluster model

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/designer6.jpg "Architecture")

- From scoring section drag assign data to cluster
- then bring convert to Dataset

- Now drag and drop Execute python script
- Replace the entire section with below

```
# imports up here can be used to
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# The entry point function MUST have two input arguments.
# If the input port is not connected, the corresponding
# dataframe argument will be None.
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1 = None, dataframe2 = None):

    # Execution logic goes here
    print(f'Input pandas.DataFrame #1: {dataframe1}')

    # If a zip file is connected to the third input port,
    # it is unzipped under "./Script Bundle". This directory is added
    # to sys.path. Therefore, if your zip file contains a Python file
    # mymodule.py you can import it using:
    # import mymodule
    Sum_of_squared_distances = []
    K = range(1,50)
    df = pd.get_dummies(dataframe1, dummy_na=True)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(df)
        Sum_of_squared_distances.append(km.inertia_)
    
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    img_file = "kmeanschart1.png"
    plt.savefig(img_file)

    from azureml.core import Run
    run = Run.get_context(allow_offline=True)
    run.upload_file(f"graphics/{img_file}", img_file)

    # Return value must be of a sequence of pandas.DataFrame
    # E.g.
    #   -  Single return value: return dataframe1,
    #   -  Two return values: return dataframe1, dataframe2
    return dataframe1,
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/designer4.jpg "Architecture")