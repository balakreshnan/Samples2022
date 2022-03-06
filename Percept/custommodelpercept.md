# Build custom model using Azure percept Device

## Process to collect images and train custom model using Azure Percept

## requirements

- Azure Account
- Azure Percept Device
- Azure Percept Studio
- Azure Cognitive custom vision service

## Steps

- First get the Azure percept device installed and configured
- Then go to - https://ms.portal.azure.com/#blade/AzureEdgeDevices/Main/overview - Azure Percept Studio
- Click the overview page
- Click New to AI models

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/Percept/images/fig1.jpg "Entire Flow")

- Then click Create a Vision Prototype

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/Percept/images/fig2.jpg "Entire Flow")

- Now give a name to the project and select resource group
- Make sure select the appropirate cognitive service custom vision resource
- Select Object detection
- For Optimization please select Accuracy

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/Percept/images/fig3.jpg "Entire Flow")

- Now lets configure the device to collect images
- Select the IoT Hub and then device
- For how often to collect i am choosing every 5 seconds take 1 frame and then send it to the custom vision service

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/Percept/images/fig4.jpg "Entire Flow")

- Next is go to custom vision and tag the images with label
- In my case i am taking images of steel ball and want to detect those for counting

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/Percept/images/fig5.jpg "Entire Flow")

- Make sure there are minimum 25 images collected
- If not then go back again and collect more images
- I almost tried 3 times to get correct amount of images
- Also remember some time we might need more than 50 or more images to get accuracy
- Here is the list of images

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/Percept/images/fig6.jpg "Entire Flow")

- Now time to detect box
- Spend time on making sure draw the bounding boxes on the images and tag them with the label

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/Percept/images/fig7.jpg "Entire Flow")

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/Percept/images/fig8.jpg "Entire Flow")

- Now click Training in custom vision
- Wait for the training to complete

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/Percept/images/fig12.jpg "Entire Flow")

- Now go back to Azure Percept Studio
- Go to next which is deploy model

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/Percept/images/fig13.jpg "Entire Flow")

- Now deploy the model
- Go to webstream and view the device camera output
- Give few mins to load the model deployed
- See if the device is able to predict the steel ball

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/Percept/images/fig14.jpg "Entire Flow")

- Small objects are always challenging to detect

- Test the model in Custom vision service
- use test1.jpg and test2.jpg images

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/Percept/images/fig15.jpg "Entire Flow")