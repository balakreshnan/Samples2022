# Azure Machine Learning using Tabula

## Using Tabula in Azure machine learning jupyter notebook

## pre-requistie

- Azure Account
- Azure Storage
- Azure machine learning Service
- Create a Azure machine learning workspace
- Notebook source - https://pypi.org/project/tabula-py/
- Using the example notebook

## Code

- Check Jave version

```
!java -version
```

- install tabula

```
!pip install -q tabula-py
```

- Check tabula version

```
import tabula

tabula.environment_info()
```

- Get the data set

```
import tabula
pdf_path = "https://github.com/chezou/tabula-py/raw/master/tests/resources/data.pdf"

dfs = tabula.read_pdf(pdf_path, stream=True)
# read_pdf returns list of DataFrames
print(len(dfs))
dfs[0]
```

- help

```
help(tabula.read_pdf)
```

- Set pages

```
# set pages option
dfs = tabula.read_pdf(pdf_path, pages=3, stream=True)
dfs[0]
```

- pass pages as string

```
# pass pages as string
tabula.read_pdf(pdf_path, pages="1-2,3", stream=True)
```

- Extract all pages

```
# extract all pages
tabula.read_pdf(pdf_path, pages="all", stream=True)
```

- now set area to parse

```
# set area option
dfs = tabula.read_pdf(pdf_path, area=[126,149,212,462], pages=2)
dfs[0]
```

- lets with new data

```
pdf_path2 = "https://github.com/chezou/tabula-py/raw/master/tests/resources/campaign_donors.pdf"

dfs = tabula.read_pdf(pdf_path2, columns=[47, 147, 256, 310, 375, 431, 504], guess=False, pages=1)
df = dfs[0].drop(["Unnamed: 0"], axis=1)
df
```

- read json

```
# read pdf as JSON
tabula.read_pdf(pdf_path, output_format="json")
```

- Convert to JSON

```
tabula.convert_into(pdf_path, "test.json", output_format="json")
!cat test.json
```

- convert to csv

```
tabula.convert_into(pdf_path, "test.tsv", output_format="tsv")
!cat test.tsv
```

```
tabula.convert_into(pdf_path, "test.csv", output_format="csv", stream=True)
!cat test.csv
```

- display

```
pdf_path3 = "https://github.com/tabulapdf/tabula-java/raw/master/src/test/resources/technology/tabula/spanning_cells.pdf"
dfs = tabula.read_pdf(
    pdf_path3,
    pages="1",
    lattice=True,
    pandas_options={"header": [0, 1]},
    area=[0, 0, 50, 100],
    relative_area=True,
    multiple_tables=False,
)
dfs[0]
```

- print output

```
template_path = "https://github.com/chezou/tabula-py/raw/master/tests/resources/data.tabula-template.json"
tabula.read_pdf_with_template(pdf_path, template_path)
```

## Conclusion

- To show tabula will work in Azure machine learning services
- To Show open source will work as is