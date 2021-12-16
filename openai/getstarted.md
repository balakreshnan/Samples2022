# Getting Started with Open AI with Azure machine learning service

## How to get started to use Open AI Api using Python with Azure Machine learning service

## Prerequistie

- Azure Account
- Azure Storage
- Azure Machine learning Service
- Azure Open AI Api account

## Create Open API trial account

- Go to https://beta.openai.com/api/signup
- Create a trial account
- Review the documentation
- For this article we are going to use python client
- on your settings grab the key

## Azure Machine Learning Service

- Login to Azure portal
- Go to Azure Machine Learning Service
- Click workspace launch
- or go to ml.azure.com
- Create or Start your compute instance
- Create a new notebook with python 3.8 with Aml SDK

- Install open ai

```
pip install openai
```

- Restart the kernel
- now write code
- First initialize the key

```
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

- now to invoke the client

```
import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = OPENAI_API_KEY

response = openai.Completion.create(engine="davinci", prompt="How is the stock market doing today?", max_tokens=50)
```

- here is the response

```
<OpenAIObject text_completion id=cmpl-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx at 0x7fbbea0396d0> JSON: {
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "logprobs": null,
      "text": "\u201d \u201cWhat\u2019s the price of gas today?\u201d and \u201cWhat would house prices be?\u201d Okay, now apply these to blockchain research. \u201cIs Bitcoin\u2019s value going up?\u201d;"
    }
  ],
  "created": 1639665062,
  "id": "cmpl-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "model": "davinci:2020-05-03",
  "object": "text_completion"
}
```