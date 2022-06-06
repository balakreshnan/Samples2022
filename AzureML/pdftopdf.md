# Process PDF's in Azure Machine Learning Notebook

## Use Azure Machine learning jupyter to process PDF to extract OCR contents

## pre-requisites

- Azure Account
- Azure Storage
- Azure Machine Learning


## Code

- Install Libraries

```
pip install pytesseract
pip install pdf2image
```

- Now go to Terminal and install poppler

```
sudo apt-get install poppler-utils
```

- Now in jupyter notebook

```
import os
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
from numpy import asarray
```

- Load the image

```
filePath = 'BOL122.PDF'
doc = convert_from_path(filePath, fmt='jpeg')
```

- Parse the image for text
- Also calulate how long does it take

```
path, fileName = os.path.split(filePath)
fileBaseName, fileExtension = os.path.splitext(fileName)
import time
start = time.process_time()

for page_number, page_data in enumerate(doc):
    #txt = pytesseract.image_to_string(Image.fromarray(page_data.values)).encode("utf-8")
    txt = pytesseract.image_to_string(Image.fromarray(asarray(page_data))).encode("utf-8")
    if('Ship-To' in txt.decode("utf-8")):
        Image.fromarray(asarray(page_data)).save('sample1.jpg')
        print('Image created \n')
    print("Page # {} - {}".format(str(page_number),txt))

print(time.process_time() - start)
```

## Done