# Translate Text from one language to another using Power Flow

## Not using AI builder just using HTTP Connector and Azure Cognitive Service Translator

## Use Case

- Create a Azure Cognitive service Translator Resource
- Using Manual Trigger
- Real time send Text to translate to multiple languages
- Using Power Flow

## Entire Process

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/translate2.jpg "Architecture")

## Steps

- First create a Power Flow with manual Trigger

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/translate3.jpg "Architecture")

- Next Add HTTP Task to get Oauth in case if you are using Azure AD authentication
- Get the Login endpoint
- Get your service principal Client ID and Client Secret
- Get the Tenant ID

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/translate4.jpg "Architecture")

- Configure the URL and other blank spaces with appropriate values
- Next Parse the JSON output to get access token
- for JSON response parse use the schema below

```
{
    "type": "object",
    "properties": {
        "token_type": {
            "type": "string"
        },
        "expires_in": {
            "type": "integer"
        },
        "ext_expires_in": {
            "type": "integer"
        },
        "access_token": {
            "type": "string"
        }
    }
}
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/translate5.jpg "Architecture")

- now send the information text to translate
- Note for the global text translation api to work URL, subscription key from Azure portal and region is important
- Other wise it will error out

```
https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&from=en&to=zh-Hans&to=de&includeSentenceLength=true&includeAlignment=true&textType=html

Content-Type: application/json
Ocp-Apim-Subscription-Key: Subscription Key from Azure Portal
Ocp-Apim-Subscription-Region: centralus (use your own region where the cognitive services resoruce was created)
```

- now the Body
- Multiple lines are allowed
- There are character limits so please check the limits page for current values

```
[
  {
    "Text": "I would really like to drive your car around the block a few times for Microsoft."
  }
]
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/translate6.jpg "Architecture")

- Next save and run

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/translate1.jpg "Architecture")

## Notes

- Custom translator has Blue score

```
https://docs.microsoft.com/en-us/azure/cognitive-services/translator/custom-translator/what-is-bleu-score
```

- To prevent certain words not getting translated

```
https://docs.microsoft.com/en-us/azure/cognitive-services/translator/prevent-translation
```