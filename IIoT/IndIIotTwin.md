# Industrial IoT with Digital Twin and Historical Data

## Using Azure Digital Twins and Azure Data Exploere to Build Industrial Platforms

## Architecture

- End to End data flow and also components

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/IIoT/images/carseatiot.jpg "Architecture")

## Components

- Azure IoT Edge Runtime (modules as docker containers)
- Azure IoT Hub - Cloud Device and Security management
- Azure Event Grid - Send telemetry to event grid
- Azure Funtions to read from Event Grid and write to Azure Digital Twin
- Azure Digital Twins - Store Current data in Twin and Export to Event Hub
- Azure Event Hub - intermediate store to push to Azure Data explorer
- Azure Data Explorer - historical data storage for further analytics
- Azure Container registery - to store images

## Setup

- Setup digital twin
- property and device id should match the iot hub device id
- Property is based on sensor tag name

### Azure IoT Hub

- Create a new IoT Hub
- Create a new IoT Edge
- install the Edge in client

### Azure IoT Edge Module

- https://docs.microsoft.com/en-us/azure/iot-edge/tutorial-python-module?view=iotedge-2020-11

- Create new module with simulated dataset
- Use Visual Studio Code
- Deploy the model to device and test
- Create a deployment for single device manifest file to deploy

### Telemetry from Iot Hub into Azure Digital Twin using azure functions

- Create a Eventgrid and function to read telemetry from Iot Hub and push into Azure Digital Twin

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/IIoT/images/adt2.jpg "Architecture")

- using visual studio 2019 to build the function

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/IIoT/images/adt1.jpg "Architecture")

- Function code has to be changed to read the messages from event grid
- Adjust the code to get the sensor data from the message and store in digital twin

- Code Repo: https://github.com/balakreshnan/ADTTwinNX

- Create a digital twin and create the property and twin

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/IIoT/images/adt3.jpg "Architecture")

- https://docs.microsoft.com/en-us/azure/digital-twins/how-to-ingest-iot-hub-data?tabs=portal

### Azure Digital Twins to Azure Data Explorer

- First create endpoints and routes in azure digital twins

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/IIoT/images/adt4.jpg "Architecture")

- Create Route

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/IIoT/images/adt5.jpg "Architecture")

- Next configure data history ingestion in Azure Digital Twin

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/IIoT/images/adt6.jpg "Architecture")

- https://docs.microsoft.com/en-us/azure/digital-twins/how-to-use-data-history?tabs=portal
- Keep an eye on persmission.
- Once everything is configured and run the sumulator in edge to send data out
- Check the Digital Twin and Azure Data Explorer for data

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/IIoT/images/adt7.jpg "Architecture")