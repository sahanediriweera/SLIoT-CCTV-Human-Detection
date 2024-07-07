# CCTV Human Detection Application

This is a Python application that uses YOLOv5 and PyTorch to detect humans in CCTV footage. The application captures video from multiple sources, processes it using a YOLOv5 model, and sends notifications when humans are detected.

## Prerequisites

Before running the application, ensure you have the following installed:

- [Python](https://www.python.org/) (v3.8 or higher)
- [PyTorch](https://pytorch.org/get-started/locally/) (v1.8 or higher)
- [OpenCV](https://opencv.org/) (v4.5 or higher)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/cctv-human-detection.git
   cd cctv-human-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download YOLOv5 weights:

   Download the YOLOv5 weights file (`yolov5s.pt`) and place it in the `weights` directory.

## Configuration

Create a `weights` directory in the root of the project and place the YOLOv5 weights file (`yolov5s.pt`) inside it.

## Usage

Run the application with the following command:

```bash
python main.py --weights weights/yolov5s.pt --source data/sample/vid.mp4 --types yolov5 --outname output
```

### Command Line Arguments

- `--weights`: Path to the model weights file.
- `--types`: Type of model to use (`yolov5` or `fast`).
- `--source`: Path to the video file or webcam URL.
- `--multisource`: List of multiple webcam URLs (optional).
- `--outname`: Name of the output file.

## Code Overview

### main.py

This is the main entry point of the application. It initializes the model, captures video from the specified source, and processes the video frames to detect humans.

```python
import argparse
from pathlib import Path
import torch
import torchvision
from torchvision import transforms as T
import cv2
import numpy as np
import os
import time
import multiprocessing

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

root_weights = ROOT/'weights/yolov5s.pt'

def send_actual_email(outname):
    print("Email sent")
    with open(f"./data/emaillog/email_log_{outname}.txt", "a") as f:
        f.write('\n')
        f.write(f'{time.time()}')

def check_log_exist(outname):
    return os.path.exists(f'./data/emaillog/email_log_{outname}.txt')

def send_email(outname):
    log_existence = check_log_exist(outname)

    if log_existence:
        with open(f'./data/emaillog/email_log_{outname}.txt', "r") as f:
            email_data = f.read()
        email_array = email_data.split("\n")
        last_time = float(email_array[-1])
        now_time = time.time()
        time_diff = now_time - last_time

        if time_diff > 3600.0:
            send_actual_email(outname)
        else:
            print("1 hour hasn't passed yet")
    else:
        with open(f'./data/emaillog/email_log_{outname}.txt', "a") as f:
            f.write(f'{time.time()}')
        send_actual_email(outname)

def write_time(outname):
    print('\a')
    local_time = time.ctime(time.time())
    with open(f'./data/Records/log_{outname}.txt', "a") as f:
        f.write('\n')
        f.write(f'{local_time},{time.time()}')

def get_prediction(img, fast_model, threshold=0.5):
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = fast_model([img])
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_box = pred_boxes[:pred_t+1]
    pred_class = pred_classes[:pred_t+1]

    return pred_box, pred_class

def run_yolov5_single_cam(weights, multisource, outname, run_on_single_cam=True):
    print(weights)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)

    if weights == root_weights:
        model.classes = [0]

    cap = cv2.VideoCapture(multisource[0])

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(os.path.join('data', 'Single Cam', f'yolov {outname}.avi'), cv2.VideoWriter_fourcc('P', 'I', 'M', '1'), fps, (width, height))
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            results = model(frame)
            detect_data = results.pandas().xyxy[0]
            presence = 'person' in detect_data["name"].tolist() if weights == root_weights else 'human' in detect_data["name"].tolist()

            if presence:
                write_time(outname)
                send_email(outname)
            frame = np.squeeze(results.render())

            video_writer.write(frame)
            if run_on_single_cam:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(e)
            break

    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()

def run_fast_single_cam(multisource, outname):
    print('fast running single cam')

    thres = 0.5
    rect_th = 1
    fast_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    fast_model.eval()
    cap = cv2.VideoCapture(multisource[0])

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')
    video_writer = cv2.VideoWriter(os.path.join('data', 'Single Cam', f'fast {outname}.avi'), fourcc, fps, (width, height))
    print("Data loaded for video capture")
    while cap.isOpened():
        ret, frame = cap.read()

        boxes, pred_clas = get_prediction(frame, fast_model, threshold=thres)
        print('Frame predictions ready')

        human_presence = 'person' in pred_clas
        if human_presence:
            write_time(outname)
            send_email(outname)

        for i in range(len(boxes)):
            if pred_clas[i] == 'person':
                x1, y1 = boxes[i][0]
                x2, y2 = boxes[i][1]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(127, 0, 255), thickness=rect_th)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):


            break

    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()

def run_yolov5(weights, multisource, outname):
    print('running yolov5')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
    if weights == root_weights:
        model.classes = [0]
    pool = []
    for source in multisource:
        pool.append(multiprocessing.Process(target=run_yolov5_single_cam, args=(weights, [source], outname, False)))
    for proc in pool:
        proc.start()
    for proc in pool:
        proc.join()

def run_fast(multisource, outname):
    print('running fast')
    pool = []
    for source in multisource:
        pool.append(multiprocessing.Process(target=run_fast_single_cam, args=([source], outname)))
    for proc in pool:
        proc.start()
    for proc in pool:
        proc.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCTV Human Detection")
    parser.add_argument("--weights", type=str, default=root_weights, help="Path to model weights")
    parser.add_argument("--types", type=str, default='yolov5', help="Model types")
    parser.add_argument("--source", type=str, default=0, help="Video source or webcam URL")
    parser.add_argument("--multisource", nargs='+', help="List of webcam URLs")
    parser.add_argument("--outname", type=str, default='output', help="Output file name")
    args = parser.parse_args()

    model_type = args.types
    source = args.source
    weights = args.weights
    outname = args.outname

    if args.multisource:
        if model_type == 'yolov5':
            run_yolov5(weights, args.multisource, outname)
        elif model_type == 'fast':
            run_fast(args.multisource, outname)
    else:
        if model_type == 'yolov5':
            run_yolov5_single_cam(weights, [source], outname)
        elif model_type == 'fast':
            run_fast_single_cam([source], outname)
```

### Key Functions

- `send_actual_email(outname)`: Sends an actual email notification.
- `send_email(outname)`: Checks the email log and sends an email if the last email was sent more than an hour ago.
- `write_time(outname)`: Writes the current time to the log file.
- `get_prediction(img, fast_model, threshold=0.5)`: Gets predictions from the Fast R-CNN model.
- `run_yolov5_single_cam(weights, multisource, outname, run_on_single_cam=True)`: Runs YOLOv5 on a single camera.
- `run_fast_single_cam(multisource, outname)`: Runs Fast R-CNN on a single camera.
- `run_yolov5(weights, multisource, outname)`: Runs YOLOv5 on multiple cameras.
- `run_fast(multisource, outname)`: Runs Fast R-CNN on multiple cameras.

## Acknowledgements

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)

Feel free to contribute and enhance the application.
 
