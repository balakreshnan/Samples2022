# Inference Azure Cognitive Custom Vision model in Jetson Xavier in ONNX

## Custom vision Compact S1 model export to ONNX

## Setup

- Azure Account
- Azure Storage
- Azure Cognitive Custom Vision Service
- I used customvision.ai web site to upload the images and create a model
- Export model as ONNX format
- Usb camera for jetson xavier nx huehd.com

``` 
Note: working with Xavier is not super easy with open source packages
```

## Setup Xavier

- I only had jetpack 4.6
- Upgrage python 3.6.9 to python 3.7.5
- Follow this tutorial - https://www.itsupportwale.com/blog/how-to-upgrade-to-python-3-7-on-ubuntu-18-10/

```
sudo apt-get install python3.7
```

- install pip

```
sudo apt install python3-pip
```

- Set alternatives

```
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
```

```
sudo update-alternatives --config python3
```

- Select 2 to set 3.7 as default

- Now check the python3 verssion

```
python3 --version
```

- Should display 3.7.5

## Now setup jetson inference

- Setup jetson inference package

```
sudo apt-get update
sudo apt-get install git cmake libpython3-dev python3-numpy
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
mkdir build
cd build
cmake ../
make
sudo make install
sudo ldconfig
```

- Follow the tutorial from there - https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md

- install Pillow, imutils, matlabplot, opencv-python after python 3.7 is installed


```
pip3 install pillow, imutils, opencv-python
sudo python3 -m pip uninstall matplotlib
sudo python3 -m pip install matplotlib
pip3 install tensorflow
pip3 install pytorch
```

- Sample URL for access Azure Cognitive Custom vision exported model - https://github.com/Azure-samples/customvision-export-samples

- now install onnx runtime

```
pip3 install onnx
pip3 install onnxruntime
```

## Export Model from Azure Cognitive Service Custom vision

- Go tho http://customvision.ai

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/jetsonxavier3.jpg "Architecture")

- Go to Export Select onnx format

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/jetsonxavier2.jpg "Architecture")

## Testing sample code with image file 

- This is to test if onnx runtime is working before we integrate into main code
- Here is the code repo - https://github.com/balakreshnan/aipinbothack/tree/main/AIPinBall/camera
  
```
python3 predictonnx.py steelball/model.onnx pinballframe8890.jpg
```

## Code to inference using Onnx in jetson xavier nx

- Here is the code repo - https://github.com/balakreshnan/aipinbothack/tree/main/AIPinBall/camera
- Now import necessary libraries
- Model path - https://github.com/balakreshnan/aipinbothack/tree/main/AIPinBall/camera/steelball

```
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils
import argparse
import pathlib
import numpy as np
import onnx
import onnxruntime
import PIL.Image
```

- Set the threshold

```
PROB_THRESHOLD = 0.40  # Minimum probably to show results.
```

- Now create the class to predict with onnx

```
class Model:
    def __init__(self, model_filepath):
        self.session = onnxruntime.InferenceSession(str(model_filepath))
        assert len(self.session.get_inputs()) == 1
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filepath)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True

    def predict(self, image_filepath):
        # image = PIL.Image.open(image_filepath).resize(self.input_shape)
	#height = image_filepath.shape[0]
	#width = image_filepath.shape[1]
        image_array = jetson.utils.cudaToNumpy(image_filepath)
        image = PIL.Image.fromarray(image_array, 'RGB').resize(self.input_shape)
        # image = PIL.Image.frombuffer("RGBX", (720,1280), image_filepath).resize(self.input_shape)
        # image = image_filepath.resize(320,320)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}


def print_outputs(outputs):
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
        if score > PROB_THRESHOLD:
            # print("{class_id}")
            print(f"Az Cog Label: {class_id}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
```

- Invoke ssd base model

```
net = detectNet("ssd-mobilenet-v2", threshold=0.5)
# net = jetson.inference.detectNet(argv=["-model=home/office/project/camera/steelball/model.onnx",  "--labels=home/office/project/camera/steelball/ labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output- bbox=boxes"], threshold=0.5)
```

- Initial onnx model

```
camera = videoSource("/dev/video0")      # '/dev/video0' for V4L2
display = videoOutput("display://0") # 'my_video.mp4' for file
model_path = "steelball/model.onnx"

model = Model(model_path)
```

- now run the model base and also created custom model

```
while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	display.Render(img)
	dimensions = img.shape
	# print(dimensions)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()) + " {:.0f} Frame Rate display".format(display.GetFrameRate()))
	print("Object Detection | N<<<<<<<<<<<<<<<<etwork {:.0f} FPS".format(net.GetNetworkFPS()))
	# for detection in detections:
	    # print(detection)
	#     print(net.GetClassDesc(detection.ClassID) + " " + str(detection.Confidence) + " " + str(detection.Left) + " " + str(detection.Top) + " " + str(detection.Right) + " " + str(detection.Bottom) + " " + str(detection.Width) + " " + str(detection.Height))
	    # print(net.GetClassDesc(detection.ClassID))
	# print("{:.0f} FPS".format(display.GetFrameRate()))
	outputs = model.predict(img)
	print_outputs(outputs)
```

- Now you should see a video source and also the terminal should display output from onnx model
- Frame runs with 4 or 5 frame rate
- SSD model can run through 70 Frame per second
- Azure Cognitive Custom Vision Onnx exported model took about 210 to 230 milliseconds for single class prediction
- Model onnx size was close to 10MB
- Below you can see the output of the model detecting the steel ball.

![Architecture](https://github.com/balakreshnan/Samples2022/blob/main/AzureAI/images/pinballmodel.png "Architecture")