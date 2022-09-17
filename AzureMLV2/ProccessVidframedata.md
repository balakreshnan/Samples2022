# Process Video and convert to frame and store as data in Azure Machine learning

## To process video and convert to frame and store as data in Azure Machine learning

## Pre-requisites

- Azure Machine Learning Workspace
- Azure Storage Account
- Video to use

## Goal of this tutorial

- To show how to take video and convert to frame
- Once the frame are in image format, create a data set in Azure Machine Learning
- upload the images
- Then we can use Data Labelling to label the images

## Code

- Make sure we have the video uploaded manually into notebook folder
- Create a notebook
- i choose python 3.10 with SDK V2
- To use the use Azure ML SDK V2

```
import cv2
import time
import os

from matplotlib import pyplot as plt
```

```
time_start = time.time()
```

- Create a function to take frame by frame and write back as jpg

```
filename = "pinballframe"

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        #cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        cv2.imwrite(output_loc + filename + "%d.jpg" % (count+1), frame)
        #cv2.imshow("Tracking", frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break
```

- Now call the above function with folder and video file name
- Provide a directory name to store the images
- I am using a Pin ball video with mp4 extension
- for input_loc change the lcoation where your video file is located
- output_loc is where the code saves individual frame as image

```
if __name__=="__main__":

    input_loc = 'pinballvideo/WIN_20220809_19_57_13_Pro.mp4'
    output_loc = 'pinballimages1/'
    video_to_frames(input_loc, output_loc)
```

## Register the data set

- Now we have to create a data set
- Upload the images to the data set

```
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
```

- Invoke the workspace ml client to connect and leverage the workspace

```
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

credential = DefaultAzureCredential()
ml_client = None
try:
    #ml_client = MLClient.from_config(credential)
    subscription_id = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    resource_group = "rgname"
    workspace = "wkspacename"
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)
except Exception as ex:
    print(ex)
    # Enter details of your AzureML workspace
    subscription_id = "xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    resource_group = "rgname"
    workspace = "wkspacename"
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)
```     

- now create the data set and upload

```
my_path = 'pinballimages1'

my_data = Data(
    path=my_path,
    type=AssetTypes.MLTABLE,
    description="Pin Ball Images for AI Hackathon",
    name="pinballimages1",
    version='1'
)

ml_client.data.create_or_update(my_data)
```

- Wait for the upload to complete
- Only 100MB file size max is uploaded.