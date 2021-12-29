# Simple Application to summarize data using GPT-3 openai model

## Let's build a Power App to summarize data using GPT-3 openai model

## What's needed

- First go to - https://beta.openai.com/ and sign up for an trial account
- If you organization has an account then register with that account
- Create a new API key
- Go to Right top in the above web site and click your name and then click on the API key
- then create a new key to use
- please make sure delete the key after completing the tutorial
- Go to documentation and take a look at completions
- we are going to use completion api
- Also need Azure account and power platform license

## Create a Power App

- To create a power app first need to create a power flow
- Flow is invoked by a powerapp trigger
- Text information will be passed to the flow

## Power Flow

- Let's create a power flow
- On the left menu in power apps click on flows
- https://make.preview.powerapps.com/
- Click on flows
- Click New Flow
- Name it as getsummary
- here is the entire flow

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/openaicompletion1.jpg "Architecture")

- First add trigger as Power Apps
- then Initialize a variable

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/openaicompletion2.jpg "Architecture")

- for value assign from Power apps
- that will take the input value and assign to the variable called prompt
- Now lets send the data to openai API to use davinci model using GPT-3
- First bring HTTP action
- Then select the action as POST
- here is the URL 

```
https://api.openai.com/v1/engines/davinci-msft/completions
```

- Note we need content-type as application/json
- also need Authorization as Bearer <your_api_key>
- here is the body

```
{
  "prompt": @{triggerBody()['Initializevariable_Value']},
  "temperature": 0.5,
  "max_tokens": 100,
  "top_p": 1,
  "frequency_penalty": 0.2,
  "presence_penalty": 0,
  "stop": [
    "\"\"\""
  ]
}
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/openaicompletion3.jpg "Architecture")

- make sure the prompt property is substituted with the value of the variable prompt as shown above
- Next we need to parse the response from above HTTP output

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/openaicompletion4.jpg "Architecture")

- Now we need to provide a sample document to parse the JSON schema

```
{
  "id": "cmpl-xxxxxxxxxxx",
  "object": "text_completion",
  "created": 1640707195,
  "model": "davinci:2020-05-03",
  "choices": [
    {
      "text": " really bright. You can see it in the sky at night.\nJupiter is the third brightest thing in the sky, after the Moon and Venus.\n",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ]
}
```

- Schema generated from sample

```
{
    "type": "object",
    "properties": {
        "id": {
            "type": "string"
        },
        "object": {
            "type": "string"
        },
        "created": {
            "type": "integer"
        },
        "model": {
            "type": "string"
        },
        "choices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string"
                    },
                    "index": {
                        "type": "integer"
                    },
                    "logprobs": {},
                    "finish_reason": {
                        "type": "string"
                    }
                },
                "required": [
                    "text",
                    "index",
                    "logprobs",
                    "finish_reason"
                ]
            }
        }
    }
}
```

- initalize a variable called outsummary

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/openaicompletion5.jpg "Architecture")
- select the Type as String

- After parsing we need to loop the array and assign the text to the variable
- Bring Apply to each action
- Select Choices as the array property
- now bring Set variable action
- Assign the currentitem to the variable outsummary

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/openaicompletion6.jpg "Architecture")

- Next add Respond to Power Apps
- Sent the outsumamry as response back to Power Apps

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/openaicompletion7.jpg "Architecture")

- Save the flow
- Do a manual test run by passing sample text
- If successful then you are set with flow

## Power Apps

- Now lets create a Power App
- This is only a simple app
- i am creating a canvas app
- Name the app as: OpenAPITest

```
Note: this process can be applied to any HTTP REST enabled actions needed to be invoked by Power Apps
```

- Now we need to create a canvas
- Bring Text Input Box
- Add default text as prompt

```
My second grader asked me what this passage means:\n\"\"\"\nJupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus.\n\"\"\"\nI rephrased it for him, in plain language a second grader can understand:\n\"\"\"\nJupiter is the fifth planet from the Sun. It is a big ball of gas. It is really bright, and you can see it in the sky at night. The ancient people named it after the Roman god Jupiter.\nJupiter is really big. It is bigger than all of the other planets in the Solar System combined. Jupiter is so big that if you could fit all of the other planets inside of Jupiter, you could still see Jupiter shining in the night sky!\nJupiter is
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/openaicompletion8.jpg "Architecture")

- Now add a button
- Call the flow and assign the return value to the variable

```
Set(summarytext,getsummary.Run(TextInput1.Text))
```

- in OnSelect apply the above formulat.
- getsummary is the name of the flow and we are passing parameters as textinput1.text
- Now lets add a Text lable as label1
- Assign the text property to summarytext.summarytext
- summarytext is the output property set in the flow

```
summarytext.summarytext
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/openaicompletion9.jpg "Architecture")

- Save the canvas app
- Run the app and test it
- below should be the output

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/openaicompletion10.jpg "Architecture")

- The above flow can be used to access most API's in open AI.
- So does we can use this for other Cognitive services