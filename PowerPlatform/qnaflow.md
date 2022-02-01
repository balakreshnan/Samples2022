# Power Flow to automate Azure Cognitive Services QnA Maker Train and Publish

## Automated flow to update KB's and Train and Publish QnA Maker

## Use Case

- Create a Qna Maker knowledge base
- Ability to update KB's based on blob trigger
- Update KB's
- Train and publish QnA Maker
- Using Power Flow

## Entire Process

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/qna6.jpg "Architecture")

## Steps

- First create a Power Flow with blob Trigger
- Configure the storage account name and use Key to authenticate

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/qna2.jpg "Architecture")

- Next bring the HTTP task to update the knowledge base
- See the Endpoint URL
- Replace KB id in xxxxxxxxxxxxxxxxxxxxxxx with actual KB id from QnA Maker UI
- Qna Maker UI Https://qnamaker.ai/

```
Method: PATCH
URL: https://qnaservicename.cognitiveservices.azure.com/qnamaker/v4.0/knowledgebases/xxxxxxxxxxxxxxxxxxxxxx
Ocp-Apim-Subscription-Key: Primary or Secondary Key from Azure Portal
Content-Type: application/json
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/qna3.jpg "Architecture")

- Here is the data that we pass

```
{
  "add": {
    "qnaList": [
      {
        "id": 0,
        "answer": "You can change the default message if you use the QnAMakerDialog. See this for details: https://docs.botframework.com/en-us/azure-bot-service/templates/qnamaker/#navtitle",
        "source": "Custom Editorial",
        "questions": [
          "How can I change the default message from QnA Maker?"
        ],
        "metadata": []
      }
    ],
    "urls": [
      "https://docs.microsoft.com/en-us/azure/cognitive-services/Emotion/FAQ"
    ],
    "files": [
      {
        "fileName": "SurfaceManual.pdf",
        "fileUri": "https://download.microsoft.com/download/2/9/B/29B20383-302C-4517-A006-B0186F04BE28/surface-pro-4-user-guide-EN.pdf"
      }
    ]
  },
  "delete": {
    "ids": [
      13,
      35
    ]
  },
  "update": {
    "name": "QnA Maker FAQ Prompts Bot",
    "qnaList": [
      {
        "id": 2,
        "answer": "You can use our REST apis to create a KB. See here for details: https://docs.microsoft.com/en-us/rest/api/cognitiveservices/qnamaker/knowledgebase/create",
        "source": "Custom Editorial",
        "questions": {
          "add": [],
          "delete": []
        },
        "metadata": {
          "add": [],
          "delete": []
        },
        "context": {
          "isContextOnly": false,
          "promptsToAdd": [
            {
              "displayText": "Add Prompts",
              "displayOrder": 0,
              "qna": {
                "id": 0,
                "answer": "Click here to know more https://docs.microsoft.com/en-us/azure/cognitive-services/qnamaker/how-to/multiturn-conversation",
                "source": "Editorial",
                "questions": [
                  "How can I add prompts?"
                ],
                "metadata": [],
                "context": {
                  "isContextOnly": false,
                  "prompts": []
                }
              },
              "qnaId": 0
            },
            {
              "displayText": "Delete Prompts",
              "displayOrder": 0,
              "qna": {
                "id": 0,
                "answer": "Click here to know more https://docs.microsoft.com/en-us/azure/cognitive-services/qnamaker/how-to/multiturn-conversation",
                "source": "Editorial",
                "questions": [
                  "How can I delete delete prompts?"
                ],
                "metadata": [],
                "context": {
                  "isContextOnly": false,
                  "prompts": []
                }
              },
              "qnaId": 0
            },
            {
              "displayText": "Update Knowledgebase",
              "displayOrder": 0,
              "qna": null,
              "qnaId": 3
            }
          ],
          "promptsToDelete": [
            3
          ]
        }
      },
      {
        "id": 3,
        "answer": "You can use our REST apis to update your KB. See here for details: https://docs.microsoft.com/en-us/rest/api/cognitiveservices/qnamaker/knowledgebase/update",
        "source": "Custom Editorial",
        "questions": {
          "add": [],
          "delete": []
        },
        "metadata": {
          "delete": [
            {
              "name": "category",
              "value": "api"
            }
          ],
          "add": [
            {
              "name": "category",
              "value": "programmatic"
            }
          ]
        },
        "context": {
          "isContextOnly": false,
          "promptsToAdd": [
            {
              "displayText": "Regenerate Endpoint keys",
              "displayOrder": 1,
              "qna": null,
              "qnaId": 4
            }
          ],
          "promptsToDelete": [
            4
          ]
        }
      }
    ]
  }
}
```

- Now lets train the KB post
- Method: POST

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/qna4.jpg "Architecture")

- There are few changes here the URL is different

```
URL: https://qnaservice.azurewebsites.net/qnamaker/knowledgebases/xxxxxxx/train
Authorization: Key from QNA Maker portal Publish screen
```

```
Method: POST
Content-Type: application/json
```

```
{
  "feedbackRecords": [
    {
      "userId": "sd53lsY=",
      "userQuestion": "qna maker with luis",
      "qnaId": 4
    }
  ]
}
```

- Publish the KB after training
- Method: POST

```
Ocp-Apim-Subscription-Key: Primary or Secondary Key from Azure Portal
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/qna5.jpg "Architecture")

- Save and run the Test
- Upload a document to blob to trigger

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/qna1.jpg "Architecture")