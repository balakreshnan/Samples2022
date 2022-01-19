# Covid 19 Data set visualize using geopandas

## How to visualize the data using geopandas

## Prerequsites

- Install geopandas
- Get the covid data from https://github.com/GoogleCloudPlatform/covid-19-open-data
- Azure Resource
- Azure Machine learning services

## Code

## Install libraries

```
pip install geopandas
```

```
pip install geoplot
```

```
pip install jupyterlab-widgets
```

```
conda install geoplot -c conda-forge
```

## Files from the data set


## Code

- Check version

```
import azureml.core
print(azureml.core.VERSION)
```

- Import libraries

```
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import numpy as np

from pandas_profiling import ProfileReport

import time

import lightgbm as lgb

from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

- Get the workspace information

```
from azureml.core import Workspace, Dataset

subscription_id = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
resource_group = 'rgname'
workspace_name = 'workspace_name'

workspace = Workspace(subscription_id, resource_group, workspace_name)
```

- First read all individual files and join them
- Vaccine world data

```
df1 = pd.read_csv('vaccinations.csv')
```

- Country specific latitude and longitude

```
df2 = pd.read_csv('geography.csv')
```

- Merge the two data sets

```
df_cd = pd.merge(df1, df2, how='left', left_on = 'location_key', right_on = 'location_key')
```

- Country specific information like names, state name and others

```
df3 = pd.read_csv('index.csv')
```

- Country specific population dataset

```
df4 = pd.read_csv('demographics.csv')
```

- GDP dataset

```
df5 = pd.read_csv('economy.csv')
```

- Merge the data sets

```
df_cd_1 = pd.merge(df_cd, df3, how='left', left_on = 'location_key', right_on = 'location_key')
```

```
df_cd_2 = pd.merge(df_cd_1, df4, how='left', left_on = 'location_key', right_on = 'location_key')
```

```
df_cd_final = pd.merge(df_cd_2, df5, how='left', left_on = 'location_key', right_on = 'location_key')
```

- Save the file

```
df_cd_final.to_csv('allcombinedcovid.csv', header=True)
```

## Geopandas code

- Read the file

```
df_cd_final = pd.read_csv('allcombinedcovid.csv')
```

```
import matplotlib.pyplot as plt
import geopandas
#from cartopy import crs as ccrs
```

```
path = geopandas.datasets.get_path('naturalearth_lowres')
```

```
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
cities = geopandas.read_file(geopandas.datasets.get_path('naturalearth_cities'))
```

```
df_cd_final['location_key'].count()
```

```
df_cd_final.drop(['Unnamed: 0'], axis = 1, inplace = True)
```

```
df_cd_final.head()
```

- Geopandas load to geo dataframe

```
gdf = geopandas.GeoDataFrame(
    df_cd_final, geometry=geopandas.points_from_xy(df_cd_final.longitude, df_cd_final.latitude))
```

- Print

```
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20,20))
ax.set_aspect('equal')
#ax = world[world.continent == 'North America'].plot(color='white', edgecolor='black')

world.boundary.plot(ax=ax, color='black')
gdf.apply(lambda x: ax.annotate(s=x.country_name, xy=x.geometry.coords[0], ha='center', fontsize=14),axis=1);
gdf.plot(ax=ax, color='red', marker='o', markersize=2)
#onlinegdf.plot(ax=ax, marker='o', color='red', markersize=2)


plt.show();
```

- More to come