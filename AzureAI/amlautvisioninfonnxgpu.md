# Inference Automated ML Vision from AML model at edge with GPU

## Onnx Azure Machine Learning Automated Vision Model inference with GPU

### Pre-requisites

- Azure Machine Learning Workspace
- Azure Storage
- Images or pictures to use


## Steps

- Create a new data labelling project for object detection
- upload the images
- Create the labels
- Label the images with single or multiple labelers
- i used the machine learning option to auto label as well
- But i had to do 350 images first
- Total of 18000 images
- Once i labelled 350, the Machine learning training started
- Once training is completed, then inference start the following day
- there were about 512 images labelled.
- Click Export and choose azure machine learning dataset and cick export
- wait for export to compelete
- Now to to Automated ML section
- Create a new experiment by selecting image dataset create above
- Select Object detection as the task
- Select yolov5, resnet50, and retinaNet as the models

## Code dependencies

- Need onnxruntime_yolov5.py
- also yolo_onnx_preprocessing_utils.py
- now the actual code here

## Code

```
import argparse
import pathlib
import numpy as np
import onnxruntime
import onnxruntime as ort
import PIL.Image
import time
import pytesseract
from PIL import Image
import cv2
import onnx
from math import sqrt
import math
import json
from yolo_onnx_preprocessing_utils import preprocess, preprocess1, frame_resize
from yolo_onnx_preprocessing_utils import non_max_suppression, _convert_to_rcnn_output
import torch
from datetime import datetime
import os
#from objdict import ObjDict
from yolo_onnx_preprocessing_utils import letterbox, non_max_suppression, _convert_to_rcnn_output
from onnxruntime_object_detection import ObjectDetection
import tempfile
#import tensorflow as tf
from torchvision import transforms
from json_tricks import dumps


PROB_THRESHOLD = 0.40  # Minimum probably to show results.

print(" Onnx Runtime : " + onnxruntime.get_device())

#labels = ['ballfail','ballinendzone','flaphit','SteelBall','zone1','zone2']
# labels = ['ballfail','ballinendzone','ballinplunge','flaphit','SteelBall','zone1','zone2']
#labels = ['BallinPlunger','EndZone','FailZone','GameOver','leftflap','RightFlap','SteelBall']

labels_file = "steelball1aml/labels.json"
onnx_model_path = "steelball1aml/model.onnx"

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

with open(labels_file) as f:
    labels = json.load(f)
print(labels)


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

    def predict1(self, image_filepath):
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
        return outputs[0]


    def print_outputs(outputs):
        assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
        for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
            if score > PROB_THRESHOLD:
                # print("{class_id}")
                # print(f"Az Cog Label: {class_id}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
            # print(f"Az Cog Label: {labels[class_id]}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
                if (class_id >= 0 and class_id <= 3):
                    print(f"Az Cog Label: {labels[class_id]}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")

def get_predictions_from_ONNX(onnx_session,img_data):
    """perform predictions with ONNX Runtime
    
    :param onnx_session: onnx model session
    :type onnx_session: class InferenceSession
    :param img_data: pre-processed numpy image
    :type img_data: ndarray with shape 1xCxHxW
    :return: boxes, labels , scores 
    :rtype: list
    """
    sess_input = onnx_session.get_inputs()
    sess_output = onnx_session.get_outputs()
    # predict with ONNX Runtime
    output_names = [ output.name for output in sess_output]
    # print(output_names)
    pred = onnx_session.run(output_names=output_names, input_feed={sess_input[0].name: img_data})
    return pred[0]

def _get_box_dims(image_shape, box):
    box_keys = ['topX', 'topY', 'bottomX', 'bottomY']
    height, width = image_shape[0], image_shape[1]

    box_dims = dict(zip(box_keys, [coordinate.item() for coordinate in box]))

    box_dims['topX'] = box_dims['topX'] * 1.0 / width
    box_dims['bottomX'] = box_dims['bottomX'] * 1.0 / width
    box_dims['topY'] = box_dims['topY'] * 1.0 / height
    box_dims['bottomY'] = box_dims['bottomY'] * 1.0 / height

    return box_dims

def _get_prediction(label, image_shape, classes):
    
    boxes = np.array(label["boxes"])
    labels = np.array(label["labels"])
    labels = [label[0] for label in labels]
    scores = np.array(label["scores"])
    scores = [score[0] for score in scores]

    bounding_boxes = []
    for box, label_index, score in zip(boxes, labels, scores):
        box_dims = _get_box_dims(image_shape, box)

        box_record = {'box': box_dims,
                      'label': classes[label_index],
                      'score': score.item()}

        bounding_boxes.append(box_record)

    return bounding_boxes


model_path = "steelball1aml/model.onnx"

model = Model(model_path)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def main():
    print("Hello World!")
    from PIL import Image
    #read the image
    #file_path = 'pinballframe8890.jpg'
    #img = Image.open('pinballframe8890.jpg')
    # define a video capture object
#vid = cv2.VideoCapture(0)

previousframe = None
prevx = 0
prevy = 0
prevw = 0
prevh = 0
movingleft = 0
movingright = 0
movingup = 0
movingdown = 0
distance = 0

vid = cv2.VideoCapture('C:\\Users\\babal\\Downloads\\WIN_20220920_11_27_37_Pro.mp4')
vid.set(cv2.CAP_PROP_FPS,90)
#ret = vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#ret = vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print('FPS ',vid.get(cv2.CAP_PROP_FPS))
print('Width ',vid.get(cv2.CAP_PROP_FRAME_WIDTH))
print('Height ',vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Fourcc' ,vid.get(cv2.CAP_PROP_FOURCC))
print('Hue' ,vid.get(cv2.CAP_PROP_HUE))
print('RGB' ,vid.get(cv2.CAP_PROP_CONVERT_RGB))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, frame = vid.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(frame)

session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

sess_input = session.get_inputs()
sess_output = session.get_outputs()
print(f"No. of inputs : {len(sess_input)}, No. of outputs : {len(sess_output)}")

for idx, input_ in enumerate(range(len(sess_input))):
    input_name = sess_input[input_].name
    input_shape = sess_input[input_].shape
    input_type = sess_input[input_].type
    print(f"{idx} Input name : { input_name }, Input shape : {input_shape}, \
    Input type  : {input_type}")  

for idx, output in enumerate(range(len(sess_output))):
    output_name = sess_output[output].name
    output_shape = sess_output[output].shape
    output_type = sess_output[output].type
    print(f" {idx} Output name : {output_name}, Output shape : {output_shape}, \
    Output type  : {output_type}")

batch, channel, height_onnx, width_onnx = session.get_inputs()[0].shape
#batch, channel, height_onnx, width_onnx

print(session.get_inputs()[0].shape)

from onnxruntime_yolov5 import initialize_yolov5
labelPath = f'steelball1aml/labels.json'
labelFile = 'labels.json'
initialize_yolov5(onnx_model_path, labelPath, 640,0.4,0.5) 

frame_size = (960, 540)
# Initialize video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# output = cv2.VideoWriter('C:\\Users\\babal\\Downloads\\output_video_from_file.mp4', fourcc, 60, frame_size, 1)
output = cv2.VideoWriter('C:\\Users\\babal\\Downloads\\output1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, frame_size)

batch_size = session.get_inputs()[0].shape
print(labels)

# Read until video is completed 
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    start = time.process_time()
    #outputs = model.predict1(frame)
    #print(outputs)
    #outputs = model.predict1(frame)
    #image = PIL.Image.fromarray(frame, 'RGB').resize(640,640)
    #assert batch_size == frame.shape[0]
    #print(session.get_inputs()[0].shape[2:])
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(img)
    #image = PIL.Image.fromarray(img, 'RGB').resize(frame.input_shape)
    #input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
    preprocessimg = preprocess1(frame)
    #convert_tensor = transforms.ToTensor()

    img = cv2.resize(frame, (640, 640))
    # convert image to numpy
    # print(session.get_inputs()[0].shape)
    x = np.array(img).astype('float32').reshape([1, channel, height_onnx, width_onnx])
    x = x / 255

    # y = preprocessimg.tolist()
    # result = get_predictions_from_ONNX(session, preprocessimg)

    #preprocessimg = frame_resize(frame)

    #result = get_predictions_from_ONNX(session, x)
    #print(result)
    h, w = frame.shape[:2]

    frame_optimized, ratio, pad_list = frame_resize(frame, 640)
    from onnxruntime_yolov5 import predict_yolov5
    result = predict_yolov5(frame_optimized, pad_list)
    predictions = result['predictions'][0]
    new_w = int(ratio[0]*w)
    new_h = int(ratio[1]*h)
    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    annotated_frame = frame_resized.copy()
    print(json.dumps(' Prediction output: '+ str(predictions), indent=1))

    #print(predictions)

    detection_count = len(predictions)
    #print(f"Detection Count: {detection_count}")

    if detection_count > 0:
        for i in range(detection_count):
            bounding_box = predictions[i]['bbox']
            tag_name = predictions[i]['labelName']
            probability = round(predictions[i]['probability'],2)
            image_text = f"{probability}%"
            color = (0, 255, 0)
            thickness = 1
            xmin = int(bounding_box["left"])
            xmax = int(bounding_box["width"])
            ymin = int(bounding_box["top"])
            ymax = int(bounding_box["height"])
            start_point = (int(bounding_box["left"]), int(bounding_box["top"]))
            end_point = (int(bounding_box["width"]), int(bounding_box["height"]))
            annotated_frame = cv2.rectangle(annotated_frame, start_point, end_point, color, thickness)
            cv2.putText(annotated_frame,tag_name + '-' + image_text,(xmin-10,ymin-10),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
            imS = cv2.resize(annotated_frame, (960, 540))
            cv2.imshow('frame', imS)


    print(" Time taken = " + str(time.process_time() - start))

    # imS = cv2.resize(img, (960, 540))

    # Display the resulting frame
    # cv2.imshow('frame', imS)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
output.release()
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

## onnxruntime_yolov5.py

```
import numpy as np
import onnxruntime as ort
from objdict import ObjDict
import time
import os
from datetime import datetime
import torch
import json
from yolo_onnx_preprocessing_utils import letterbox, non_max_suppression, _convert_to_rcnn_output

providers = [
    'CUDAExecutionProvider',
    'CPUExecutionProvider',
]

class ONNXRuntimeObjectDetection():

    def __init__(self, model_path, classes, target_dim, target_prob, target_iou):
        self.target_dim = target_dim
        self.target_prob = target_prob
        self.target_iou = target_iou
        
        self.device_type = ort.get_device()
        print(f"ORT device: {self.device_type}")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.sess_input = self.session.get_inputs()
        self.sess_output = self.session.get_outputs()
        print(f"No. of inputs : {len(self.sess_input)}, No. of outputs : {len(self.sess_output)}")

        for idx, input_ in enumerate(range(len(self.sess_input))):
            input_name = self.sess_input[input_].name
            input_shape = self.sess_input[input_].shape
            input_type = self.sess_input[input_].type
            print(f"{idx} Input name : { input_name }, Input shape : {input_shape}, \
            Input type  : {input_type}")  

        for idx, output in enumerate(range(len(self.sess_output))):
            output_name = self.sess_output[output].name
            output_shape = self.sess_output[output].shape
            output_type = self.sess_output[output].type
            print(f" {idx} Output name : {output_name}, Output shape : {output_shape}, \
            Output type  : {output_type}")


        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        batch, channel, height_onnx, width_onnx = self.session.get_inputs()[0].shape
        self.batch = batch
        self.channel = channel
        self.height_onnx = height_onnx
        self.width_onnx = width_onnx
    
        self.classes = classes
        self.num_classes = len(classes)
             
    def predict(self, pp_image, pad_list):
        inputs = pp_image
        # predict with ONNX Runtime
        output_names = [output.name for output in self.sess_output]
        outputs = self.session.run(output_names=output_names, input_feed={self.input_name: inputs})
        filtered_outputs = non_max_suppression(torch.from_numpy(outputs[0]), conf_thres = self.target_prob, iou_thres = self.target_iou)

        def _get_box_dims(image_shape, box):
            box_keys = ['left', 'top', 'width', 'height']
            height, width = image_shape[0], image_shape[1]

            bbox = dict(zip(box_keys, [coordinate.item() for coordinate in box]))

            return bbox
        
        def _get_prediction(label, image_shape, classes):
            
            boxes = np.array(label["boxes"])
            labels = np.array(label["labels"])
            labels = [label[0] for label in labels]
            scores = np.array(label["scores"])
            scores = [score[0] for score in scores]

            pred = []
            for box, label_index, score in zip(boxes, labels, scores):
                box_dims = _get_box_dims(image_shape, box)

                prediction = {  
                    'probability': score.item(),
                    'labelId': label_index.item(),
                    'labelName': classes[label_index],
                    'bbox': box_dims
                }
                pred.append(prediction)

            return pred

        ttl_preds = []
        for result_i, pad in zip(filtered_outputs, pad_list):
            label, image_shape = _convert_to_rcnn_output(result_i, self.height_onnx, self.width_onnx, pad)
            ttl_preds.append(_get_prediction(label, image_shape, self.classes))

        if len(ttl_preds) > 0:
            # print(json.dumps(ttl_preds, indent=1))
            return ttl_preds
        else:
            print("No predictions passed the threshold")  
            return []

def log_msg(msg):
    print("{}: {}".format(datetime.now(), msg))

def checkModelExtension(fp):
  ext = os.path.splitext(fp)[-1].lower()
  if(ext != ".onnx"):
    raise Exception(fp, "is an unknown file format. Use the model ending with .onnx format")
  if not os.path.exists(fp):
    raise Exception("[ ERROR ] Path of the onnx model file is Invalid")

def initialize_yolov5(model_path, labels_path, target_dim, target_prob, target_iou):
    print('Loading labels...\n', end='')
    checkModelExtension(model_path)
    with open(labels_path) as f:
        classes = json.load(f)    
    print('Loading model...\n', end='')
    global ort_model
    ort_model = ONNXRuntimeObjectDetection(model_path, classes, target_dim, target_prob, target_iou)
    print('Success!')

def predict_yolov5(img_data, pad_list):
    log_msg('Predicting image')

    t1 = time.time()
    predictions = ort_model.predict(img_data, pad_list)
    t2 = time.time()
    t_infer = (t2-t1)
    response = {
        'created': datetime.utcnow().isoformat(),
        'inference_time': t_infer,
        'predictions': predictions
        }
    return response

def warmup_image(batch_size, warmup_dim):
    for _ in range(batch_size):
        yield np.zeros([warmup_dim, warmup_dim, 3], dtype=np.uint8)
```

## yolo_onnx_preprocessing_utils.py

```
import cv2
import numpy as np
import torch
import time
import torchvision
from PIL import Image
from typing import Any, Dict, List


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
):
    """Resize image to a 32-pixel-multiple rectangle
    https://github.com/ultralytics/yolov3/issues/232

    :param img: an image
    :type img: <class 'numpy.ndarray'>
    :param new_shape: target shape in [height, width]
    :type new_shape: <class 'int'>
    :param color: color for pad area
    :type color: <class 'tuple'>
    :param auto: minimum rectangle
    :type auto: bool
    :param scaleFill: stretch the image without pad
    :type scaleFill: bool
    :param scaleup: scale up
    :type scaleup: bool
    :return: letterbox image, scale ratio, padded area in (width, height) in each side
    :rtype: <class 'numpy.ndarray'>, <class 'tuple'>, <class 'tuple'>
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def clip_coords(boxes, img_shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)

    :param boxes: bbox
    :type boxes: <class 'torch.Tensor'>
    :return: img_shape: image shape
    :rtype: img_shape: <class 'tuple'>: (height, width)
    """
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def unpad_bbox(boxes, img_shape, pad):
    """Correct bbox coordinates by removing the padded area from letterbox image

    :param boxes: bbox absolute coordinates from prediction
    :type boxes: <class 'torch.Tensor'>
    :param img_shape: image shape
    :type img_shape: <class 'tuple'>: (height, width)
    :param pad: pad used in letterbox image for inference
    :type pad: <class 'tuple'>: (width, height)
    :return: (unpadded) image height and width
    :rtype: <class 'tuple'>: (height, width)
    """
    dw, dh = pad
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    img_width = img_shape[1] - (left + right)
    img_height = img_shape[0] - (top + bottom)

    if boxes is not None:
        boxes[:, 0] -= left
        boxes[:, 1] -= top
        boxes[:, 2] -= left
        boxes[:, 3] -= top
        clip_coords(boxes, (img_height, img_width))

    return img_height, img_width


def _convert_to_rcnn_output(output, height, width, pad):
    # output: nx6 (x1, y1, x2, y2, conf, cls)
    rcnn_label: Dict[str, List[Any]] = {"boxes": [], "labels": [], "scores": []}

    # Adjust bbox to effective image bounds
    img_height, img_width = unpad_bbox(
        output[:, :4] if output is not None else None, (height, width), pad
    )

    if output is not None:
        rcnn_label["boxes"] = output[:, :4]
        rcnn_label["labels"] = output[:, 5:6].long()
        rcnn_label["scores"] = output[:, 4:5]

    return rcnn_label, (img_height, img_width)


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right

    :param x: bbox coordinates in [x center, y center, w, h]
    :type x: <class 'numpy.ndarray'> or torch.Tensor
    :return: new bbox coordinates in [x1, y1, x2, y2]
    :rtype: <class 'numpy.ndarray'> or torch.Tensor
    """
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    """Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    :param box1: bbox in (Tensor[N, 4]), N for multiple bboxes and 4 for the box coordinates
    :type box1: <class 'torch.Tensor'>
    :param box2: bbox in (Tensor[M, 4]), M is for multiple bboxes
    :type box2: <class 'torch.Tensor'>
    :return: iou of box1 to box2 in (Tensor[N, M]), the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    :rtype: <class 'torch.Tensor'>
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    return inter / (
        area1[:, None] + area2 - inter
    )  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(
    prediction,
    conf_thres=0.1,
    iou_thres=0.6,
    multi_label=False,
    merge=False,
    classes=None,
    agnostic=False,
):
    """Performs per-class Non-Maximum Suppression (NMS) on inference results

    :param prediction: predictions
    :type prediction: <class 'torch.Tensor'>
    :param conf_thres: confidence threshold
    :type conf_thres: float
    :param iou_thres: IoU threshold
    :type iou_thres: float
    :param multi_label: enable to have multiple labels in each box?
    :type multi_label: bool
    :param merge: Merge NMS (boxes merged using weighted mean)
    :type merge: bool
    :param classes: specific target class
    :type classes:
    :param agnostic: enable class agnostic NMS?
    :type agnostic: bool
    :return: detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    :rtype: <class 'list'>
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # min_wh = 2
    max_wh = 4096  # (pixels) maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    if multi_label and nc < 2:
        multi_label = False  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True
                )  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except Exception:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(
                    "[WARNING: possible CUDA error ({} {} {} {})]".format(
                        x, i, x.shape, i.shape
                    )
                )
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def _read_image(ignore_data_errors: bool, image_url: str, use_cv2: bool = False):
    try:
        if use_cv2:
            # cv2 can return None in some error cases
            img = cv2.imread(image_url)  # BGR
            if img is None:
                print("cv2.imread returned None")
            return img
        else:
            image = Image.open(image_url).convert("RGB")
            return image
    except Exception as ex:
        if ignore_data_errors:
            msg = "Exception occurred when trying to read the image. This file will be ignored."
            print(msg)
        else:
            print(str(ex), has_pii=True)
        return None


def preprocess(image_url, img_size=640):
    img0 = _read_image(
        ignore_data_errors=False, image_url=image_url, use_cv2=True
    )  # cv2.imread(image_url)  # BGR
    if img0 is None:
        return image_url, None, None

    img, ratio, pad = letterbox(img0, new_shape=img_size, auto=False, scaleup=False)

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img = np.ascontiguousarray(img)
    np_image = torch.from_numpy(img)
    np_image = np.expand_dims(np_image, axis=0)
    np_image = np_image.astype(np.float32) / 255.0
    return np_image, pad

def preprocess1(img0, img_size=640):

    img, ratio, pad = letterbox(img0, new_shape=img_size, auto=False, scaleup=False)
    # print('pad=',pad)

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img = np.ascontiguousarray(img)
    np_image = torch.from_numpy(img)
    np_image = np.expand_dims(np_image, axis=0)
    np_image = np_image.astype(np.float32) / 255.0
    return np_image, pad

def frame_resize(img, target=640):
    img_processed_list = []
    pad_list = []
    batch_size = 1
    for i in range(batch_size):
        img0, ratio, pad = letterbox(img, new_shape=target, auto=False, scaleup=False)
        img0 = img0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        img0 = np.ascontiguousarray(img0)
        np_image = torch.from_numpy(img0)
        np_image = np.expand_dims(np_image, axis=0)
        np_image = np_image.astype(np.float32) / 255.0
        img_processed_list.append(np_image)
        pad_list.append(pad)
    if len(img_processed_list) > 1:
        img_data = np.concatenate(img_processed_list)
    elif len(img_processed_list) == 1:
        img_data = img_processed_list[0]
    else:
        img_data = None
    assert batch_size == img_data.shape[0]
    return img_data, ratio, pad_list
```

## Conclusion

- Finalize the code