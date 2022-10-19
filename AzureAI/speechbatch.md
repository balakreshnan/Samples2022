# Processing Speech to text in batch using logic apps and Azure Cognitive Speech service

## Batch processing audio to text transcription

## Requirements

- Azure Storage
- Sample audio files
- Azure Cognitive Services - Speech to text
- Audio file container SAS key with Read and list
- Output container to store json output with read, write and list permission in SAS key
- Upload the sample audio files to container

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/batchspeech1.jpg "Architecture")

- Here is when the input files are in the container

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/batchspeech2.jpg "Architecture")

```
Note: create the sas key for input container with read and list permission, otherwise you will get invalid container URI error
```

- Create a logic app to process the audio files
- Trigger the logic app with Blob connector when any new file is added to the container
- here is the overall process

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/batchspeech4.jpg "Architecture")

- Now configure the blob storage account with connection string
- Also specify the input container

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/batchspeech5.jpg "Architecture")

- Now we need bring HTTP actvitity to call the speech to text API

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/batchspeech6.jpg "Architecture")

- Use Method as POST
- URI: https://<region>.api.cognitive.microsoft.com/speechtotext/v3.0/transcriptions
- now for headers
- Ocp-Apim-Subscription-Key: key from the cognitive service account
- Content-Type: application/json
- Now the Body

```
{
  "contentContainerUrl": "https://storage.blob.core.windows.net/callcenterinput?sp=rl&st=2022xxxxxxxxxxxxxxxxxx",
  "displayName": "audiobatch1",
  "locale": "en-US",
  "model": null,
  "properties": {
    "destinationContainerUrl": "https://storagename.blob.core.windows.net/callcenterbatchtext?sp=rwl&st=2022xxxxxxxxxxxxxxxxxxxxxx",
    "wordLevelTimestampsEnabled": true
  }
}
```

- I am providing storage account to store the output json files
- In this was we can process large volume of files
- Next is to Parse the JSON output

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/batchspeech7.jpg "Architecture")

- Make sure the body from above is selected as input
- to parse the json here is the schema

```
{
    "properties": {
        "createdDateTime": {
            "type": "string"
        },
        "displayName": {
            "type": "string"
        },
        "lastActionDateTime": {
            "type": "string"
        },
        "links": {
            "properties": {
                "files": {
                    "type": "string"
                }
            },
            "type": "object"
        },
        "locale": {
            "type": "string"
        },
        "model": {
            "properties": {
                "self": {
                    "type": "string"
                }
            },
            "type": "object"
        },
        "properties": {
            "properties": {
                "channels": {
                    "items": {
                        "type": "integer"
                    },
                    "type": "array"
                },
                "diarizationEnabled": {
                    "type": "boolean"
                },
                "profanityFilterMode": {
                    "type": "string"
                },
                "punctuationMode": {
                    "type": "string"
                },
                "wordLevelTimestampsEnabled": {
                    "type": "boolean"
                }
            },
            "type": "object"
        },
        "self": {
            "type": "string"
        },
        "status": {
            "type": "string"
        }
    },
    "type": "object"
}
```

- now use the transcription api to get the status of the transcription
- Use the parse output second self variable from parse JSON output
- Provide the subscription key
  
  ![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/batchspeech8.jpg "Architecture")

  - Output from above should be running. If failed check for the error message

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/batchspeech8.jpg "Architecture")

- now check the output in the destination container

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/batchspeech3.jpg "Architecture")
