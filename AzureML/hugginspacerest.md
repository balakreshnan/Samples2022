# Hugging Face consuming rest API

## how to consume Hugging Face's rest API

## Steps

- First deploy  the model based on documentation

```
import json
import requests
headers = {"Authorization": f"Bearer xxxxxxxx"}
API_URL = "https://xxxxxx.northcentralus.inference.ml.azure.com/score"
def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))
data = query({"inputs": "The answer to the universe is mask."})
```