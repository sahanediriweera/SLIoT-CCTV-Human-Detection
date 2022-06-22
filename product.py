import argparse
from pyexpat import model
import string
from cv2 import VideoWriter
import torch
import torchvision
from torchvision import transforms as T
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from os import listdir, path
import multiprocessing
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

root_weights = ROOT/'weights/yolov5s.pt'

def write_time():
    print('\a')
    seconds = time.time()
    local_time = time.ctime(seconds)
    f = open('./data/Records/log.txt',"a")
    f.write('\n')
    f.write('{},{}'.format(local_time,seconds))
    f.close()

def get_prediction(img,fast_model,threshold=0.5):
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
    pred_boxes = [[(i[0],i[1]),(i[2],i[3])]for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x> threshold][-1]
    pred_box = pred_boxes[:pred_t+1]
    pred_class = pred_classes[:pred_t+1]

    return pred_box, pred_class

def run_yolov5_single_cam(weights,multisource,outname,run_on_single_cam = True):
    print(weights)
    model = torch.hub.load('ultralytics/yolov5','custom',path = weights,force_reload=True)

    if weights == root_weights:
        model.classes = [0]

    cap = cv2.VideoCapture(multisource[0])

    width = int(cap. get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(cap. get(cv2.CAP_PROP_FRAME_HEIGHT ))
    fps = int(cap. get(cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(os.path.join('data','Single Cam','yolov {}.avi'.format(outname)),cv2.VideoWriter_fourcc('P','I','M','1'),fps,(width,height))
    while cap.isOpened():
        try:

            ret, frame = cap.read()
            results = model(frame)
            detect_data = results.pandas().xyxy[0]
            if weights == root_weights:
                presence = 'person'in detect_data["name"].tolist()
            else:
                presence = 'human' in detect_data["name"].tolist()
            
            if presence:
                write_time()
            frame = np.squeeze(results.render())
            
            video_writer.write(frame)
            if run_on_single_cam:
                cv2.imshow('Frame',frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        except:
            break

    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()
    

def run_fast_single_cam(multisource,outname):
    print('fast running single cam')

    thres = 0.5
    rect_th = 1
    fast_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    fast_model.eval()
    cap = cv2.VideoCapture(multisource[0])

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc =cv2.VideoWriter_fourcc('P','I','M','1')
    #video_writer = cv2.VideoWriter(os.path.join('data','output.mp4'),fourcc,fps,(width,height))
    video_writer = cv2.VideoWriter(os.path.join('data','Single Cam','fast {}.avi'.format(outname)),fourcc,fps,(width,height))
    print("Data Loaded for video capture")
    while cap.isOpened():
        ret, frame = cap.read()

        boxes, pred_clas = get_prediction(frame,fast_model,threshold=thres)
        print('Frame predictions ready')

        for i in range(len(boxes)):
            if pred_clas[i] == 'person':
                x1,y1 = boxes[i][0]
                x2,y2 = boxes[i][1]
                x1 = int(x1)
                x2 = int(x2) 
                y1 = int(y1)
                y2 = int(y2)
                r, g, b = (127, 0, 255)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color=(r, g, b), thickness=rect_th)
        cv2.imshow('Frame',frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
                break


        video_writer.write(frame)
    video_writer.release()
    cap.release()

def run_yolov5_multi_cam(weights,multisource):
    for i in enumerate(multisource):
        j = [i]
        p = multiprocessing.Process(target=run_yolov5_single_cam(weights,i,str(i),run_on_single_cam=False,))
        p.start()

def run_fast_multi_cam(multisource):
    for i in enumerate(multisource):
        j = [i]
        p = multiprocessing.Process(target=run_fast_single_cam(j,str(i),))
        p.start()

def run_yolov5_detect_video(weights,source,outname):
    list_dir  = listdir(source)
    source = str(source)
    list_dir = str(list_dir[0])
    source = '{}\{}'.format(source,list_dir)
    model = torch.hub.load('ultralytics/yolov5','custom',path = weights,force_reload=True)

    if weights == root_weights:
        model.classes = [0]
    cap = cv2.VideoCapture(source)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_writer = cv2.VideoWriter(os.path.join('data','Recordings','yolov {}.avi'.format(outname)),cv2.VideoWriter_fourcc('P','I','M','1'),fps,(width,height))

    while cap.isOpened():
        try:

            ret, frame = cap.read()
            results = model(frame)
            cv2.imshow('Frame',np.squeeze(results.render()))
            video_writer.write(np.squeeze(results.render()))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        except:
            break

    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()
    

def run_fast_detect_video(source,outname):
    print('fast running here')
    list_dir  = listdir(source)
    source = str(source)
    list_dir = str(list_dir[0])
    source = '{}\{}'.format(source,list_dir)    

    thres = 0.5
    rect_th = 1
    fast_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    fast_model.eval()
    cap = cv2.VideoCapture(source)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc =cv2.VideoWriter_fourcc('P','I','M','1')
    now_frame = 0
    total_fps = 30*116
    #video_writer = cv2.VideoWriter(os.path.join('data','Recordings','fast {}.mp4'.format(outname)),fourcc,fps,(width,height))
    video_writer = cv2.VideoWriter(os.path.join('data','Recordings','fast {}.avi'.format(outname)),fourcc,fps,(width,height))
    while cap.isOpened():
        ret, frame = cap.read()

        boxes, pred_clas = get_prediction(frame,fast_model,threshold=thres)
        for i in range(len(boxes)):
            if pred_clas[i] == 'person':
                x1,y1 = boxes[i][0]
                x2,y2 = boxes[i][1]
                x1 = int(x1)
                x2 = int(x2) 
                y1 = int(y1)
                y2 = int(y2)
                r, g, b = (127, 0, 255) # Random Color
                cv2.rectangle(frame, (x1,y1), (x2,y2), color=(r, g, b), thickness=rect_th)
                #cv2.putText(frame, pred_clas[i], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, text_size, (r, g, b), thickness=text_th)
        cv2.imshow("Frame",frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        video_writer.write(frame)
    video_writer.release()
    cap.release()



def single_cam(types,weights,multisource,outname):
    if types == 'yolov5':
        run_yolov5_single_cam(weights,multisource,outname)
    if types == 'fast':
        run_fast_single_cam(multisource,outname)

def multi_cam(types,weights,multisource):
    if types == 'yolov5':
        run_yolov5_multi_cam(weights,multisource)
    if types == 'fast':
        run_fast_multi_cam(multisource)

def detect_video(types,weights,source,outname):
    if types == 'yolov5':
        run_yolov5_detect_video(weights,source,outname)
    if types == 'fast':
        run_fast_detect_video(source,outname)

@torch.no_grad()
def run(weights = ROOT/'weights/yolov5s.pt',source = "data/sample/vid.mp4",types='yolov5',multisource=None,outname = 'output'):
    if multisource == None:
        detect_video(types,weights,source,outname)
    else:
        if len(multisource) == 1:
            single_cam(types,weights,multisource,outname)

        if len(multisource)>1:
            multi_cam(types,weights,multisource)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=ROOT / 'weights/yolov5s.pt', help='model path(s)')
    parser.add_argument('--types',type=str,default='yolov5',help='type of model')
    parser.add_argument('--source', type=str, default=ROOT / 'data/sample', help='add urls or webcam')
    parser.add_argument('--multisource', nargs='+', type=int, help='enter multiple webcams')
    parser.add_argument('--outname',default='output',help='the name output of the name of the file')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**(vars(opt)))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

    