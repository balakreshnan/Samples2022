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
# The script MUST contain a function named azureml_main
# which is the entry point for this module.

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
    
    
    fig, axs = plt.subplots(figsize=(12, 4))        # Create an empty matplotlib Figure and Axes
    # dataframe1.plot.area(ax=axs)                   # Use pandas to put the area plot on the prepared Figure/Axes
    dataframe1.plot.box(ax=axs)  
    img_file = "boxplot.png"        # Do any matplotlib customization you like
    fig.savefig(img_file) 
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

# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to
import pandas as pd
import numpy as np
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
    
    #Initialize the class object
    kmeans = KMeans(n_clusters= 10)
    
    #predict the labels of clusters.
    label = kmeans.fit_predict(df)
    #Getting the Centroids
    centroids = kmeans.cluster_centers_
    u_labels = np.unique(label)
    print("label = ", label)
    print("labels: " , u_labels)
    print("Centroids: " , centroids)
    
    #for i in u_labels:
    #    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)


    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    plt.legend()
    plt.show()
    img_file = "kmeanschartcentroid.png"
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

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/designer7.jpg "Architecture")

- Execute python script for dataframe statistics
- Replace with below code

```
# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to
import numpy as np
import pandas as pd
#from pandas_profiling import ProfileReport

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
    print(" \nDescribe stats : \n\n", dataframe1.describe())
    print(" \nGroup by aggre : \n\n", dataframe1.agg(
        {
            "Col5": ["min", "max", "median", "skew"],
            "Col2": ["min", "max", "median", "mean"],
        }
    ))
    print(" \nCount median value count for col5, Col2 : \n\n", dataframe1[["Col5", "Col2"]].median())
    print(" \nCount mean value count for col5, Col2 : \n\n", dataframe1[["Col5", "Col2"]].mean())
    print(" \nCount distinct value count for col4 : \n\n", dataframe1["Col4"].value_counts())
    # Count total NaN at each column in a DataFrame
    print(" \nCount total NaN at each column in a DataFrame : \n\n", dataframe1.isnull().sum())

    percent_missing = dataframe1.isnull().sum() * 100 / len(dataframe1)
    missing_value_df = pd.DataFrame({'column_name': dataframe1.columns,
                                    'percent_missing': percent_missing})
    print(missing_value_df)
    # Return value must be of a sequence of pandas.DataFrame
    # E.g.
    #   -  Single return value: return dataframe1,
    #   -  Two return values: return dataframe1, dataframe2
    return dataframe1,
```

```
Describe stats : 

               Col2          Col5  ...        Col18        Col21
count  1000.000000   1000.000000  ...  1000.000000  1000.000000
mean     20.903000   3271.258000  ...     1.155000     1.300000
std      12.058814   2822.736876  ...     0.362086     0.458487
min       4.000000    250.000000  ...     1.000000     1.000000
25%      12.000000   1365.500000  ...     1.000000     1.000000
50%      18.000000   2319.500000  ...     1.000000     1.000000
75%      24.000000   3972.250000  ...     1.000000     2.000000
max      72.000000  18424.000000  ...     2.000000     2.000000

[8 rows x 8 columns]
 
Group by aggre : 

                 Col5    Col2
max     18424.000000  72.000
mean             NaN  20.903
median   2319.500000  18.000
min       250.000000   4.000
skew        1.949628     NaN
 
Count median value count for col5, Col2 : 

 Col5    2319.5
Col2      18.0
dtype: float64
 
Count mean value count for col5, Col2 : 

 Col5    3271.258
Col2      20.903
dtype: float64
 
Count distinct value count for col4 : 

 A43     280
A40     234
A42     181
A41     103
A49      97
A46      50
A45      22
A44      12
A410     12
A48       9
Name: Col4, dtype: int64
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureML/images/designer8.jpg "Architecture")

- Drag and drop Summarize Data
- Will generate statistics for dataframe