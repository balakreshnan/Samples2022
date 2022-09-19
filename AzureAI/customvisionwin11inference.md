# Inference Azure Cognitive Custom Vision model in windows 11 in ONNX

## Custom vision Compact S1 model export to ONNX

## Setup

- Azure Account
- Azure Storage
- Azure Cognitive Custom Vision Service
- I used customvision.ai web site to upload the images and create a model
- Export model as ONNX format
- Usb camera for windows 11 nx huehd.com

## Code

- Make sure you have a conda environment created with python 3.9
- install tensorflow, pytorch

- install onnxruntime-gpu
- install pillow
- install onnx
- install opencv-python

```
pip install onnxruntime-gpu
pip install onnx
pip install pillow
pip install opencv-python
```

## Python Code

- I am using GPU in my windows 11 machine
- Usb camera device is 1

```
import argparse
import pathlib
import numpy as np
import onnxruntime
import PIL.Image
import time
import pytesseract
from PIL import Image
import cv2
import onnx

PROB_THRESHOLD = 0.40  # Minimum probably to show results.

print(" Onnx Runtime : " + onnxruntime.get_device())

labels = ['ballfail','ballinendzone','flaphit','SteelBall','zone1','zone2']

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]

class Model:
    def __init__(self, model_filepath):
        self.session = onnxruntime.InferenceSession(str(model_filepath), providers=providers)
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
        #image = PIL.Image.open(image_filepath).resize(self.input_shape)
	    #height = image_filepath.shape[0]
	    #width = image_filepath.shape[1]
        #image_array = jetson.utils.cudaToNumpy(image_filepath)
        #image = PIL.Image.fromarray(image_array, 'RGB').resize(self.input_shape)
        # image = PIL.Image.frombuffer("RGBX", (720,1280), image_filepath).resize(self.input_shape)
        # image = image_filepath.resize(320,320)
        img = cv2.cvtColor(image_filepath, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(img)
        image = PIL.Image.fromarray(img, 'RGB').resize(self.input_shape)
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
                # print(f"Az Cog Label: {class_id}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
            # print(f"Az Cog Label: {labels[class_id]}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
                if (class_id >= 0 and class_id <= 3):
                    print(f"Az Cog Label: {labels[class_id]}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")

model_path = "steelball1/model.onnx"

model = Model(model_path)

def main():
    print("Hello World!")
    from PIL import Image
    #read the image
    #file_path = 'pinballframe8890.jpg'
    #img = Image.open('pinballframe8890.jpg')
    # define a video capture object
vid = cv2.VideoCapture(1)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    start = time.process_time()
    outputs = model.predict(frame)
    assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
    for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
	    if score > PROB_THRESHOLD:
	        if (class_id >= 0 and class_id <= 3):
                    print(f"Az Cog Label: {labels[class_id]}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
                    x = np.int32(box[0] * frame.shape[1])
                    y = np.int32(box[1] * frame.shape[0])
                    w = np.int32(box[2] * frame.shape[1])
                    h = np.int32(box[3] * frame.shape[0])
                    point_one = (x,y)
                    point_two = (x + w, y + h)
	                # img1 = cv2.rectangle(jetson.utils.cudaToNumpy(img), point_one, point_two, color=(255,211,67), thickness=2)
                    cv2.rectangle(frame, point_one, point_two, color=(255,211,67), thickness=2)
	            #jetson.utils.cudaDrawRect(img, (x, y, w, h), (255,127,0,200))
	            # print(x, y, w, h)
	            # print(f"box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3pip ]:.5f})")
    print(" Time taken = " + str(time.process_time() - start))
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- Inferencing average about 003 to 0.05 seconds per frame.