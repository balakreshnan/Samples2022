# Detect custom objects in an image or photo using custom vision ai model

## Using Azure cognitive custom vision service with power apps and flow

## Use case

- Create a physical securitya and privacy device detection
- Objects like knife, gun, etc.
- Objects like listening devices like alexa, google home, cellphone, etc.

## Create custom model

- Create a custom model
- use Azure cognitive services
- Create a Azure custom vision cognitive services account
- Collect images for classes
   - guns
   - knife
   - alexa
   - google home
   - cellphone
- Above are the objects we choose to detect, but we can add more as well
- go to https://customvision.ai
- Create a object detection project
- Create Tag

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect1.jpg "Architecture")

- Upload the images

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect2.jpg "Architecture")

- Do the bounding box for each object and assign tags

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect3.jpg "Architecture")
![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect4.jpg "Architecture")
![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect6.jpg "Architecture")

- Now Click Train button
- Click Quick Training
- Wait for 20 minutes

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect7.jpg "Architecture")

- Click on prediction url.

## Create a new Power app for detecting objects

- Now time to create Power Apps
- Go to https://make.preview.powerapps.com/
- Also get power app premium license
- Create a new Canvas app

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect8.jpg "Architecture")

- Create a blank app
- insert Media -> Add Picture -> and drag to canvas
- Click actions -> Power automate and select the flow created below
- Create the power flow as below first before proceeding to next step
- Once the flow is create then follow the below
- Insert a Button
- Insert a text box

```
Set(JSONImageSample, JSON(UploadedImage2.Image, JSONFormat.IncludeBinaryData));
Set(outputtext,getcustomvisionimage.Run(JSONImageSample, 0, 0,0));
```

- Now set the Text box default

```
outputtext.output
```

## Power flow to inference the objects

- Overall flow

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect9.jpg "Architecture")

- Bring initialive a variable

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect10.jpg "Architecture")

```
json(triggerBody()['HTTP_Body'])
```

- Now bring HTTP

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect11.jpg "Architecture")

- Now get the URL from prediction URL from custom vision
- Get Prediction-Key
- Set the content-type

```
https://cogsvcname-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/xxxxxxxxxxxxxxxxxxxxxx/detect/iterations/Iteration2/image
```

```
Content-Type: application/octet-stream
Prediction-key: xxxxxxxxxxxxxxxxxxxxx
```

- For Body select the above variable initialized
- Now bring Parse JSON

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect12.jpg "Architecture")

```
{
    "type": "object",
    "properties": {
        "id": {
            "type": "string"
        },
        "project": {
            "type": "string"
        },
        "iteration": {
            "type": "string"
        },
        "created": {
            "type": "string"
        },
        "predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "probability": {
                        "type": "number"
                    },
                    "tagId": {
                        "type": "string"
                    },
                    "tagName": {
                        "type": "string"
                    },
                    "boundingBox": {
                        "type": "object",
                        "properties": {
                            "left": {
                                "type": "number"
                            },
                            "top": {
                                "type": "number"
                            },
                            "width": {
                                "type": "number"
                            },
                            "height": {
                                "type": "number"
                            }
                        }
                    }
                },
                "required": [
                    "probability",
                    "tagId",
                    "tagName",
                    "boundingBox"
                ]
            }
        }
    }
}
```

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect13.jpg "Architecture")

- INitial variable
- Apply Each Task
- Select predictions

```
predictions
```

- Bring compose

```
{
 "tagName" : @{items('Apply_to_each')?['tagName']},
 "probability" : @{items('Apply_to_each')?['probability']}
}
```

- Append a variable
- Set output variable with above variable
- Respond to Power App
- Below is the sample output from power app

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect14.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect15.jpg "Architecture")

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/PowerPlatform/images/ppdetect16.jpg "Architecture")