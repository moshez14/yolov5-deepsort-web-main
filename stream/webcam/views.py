from django.shortcuts import render
from django.http import StreamingHttpResponse
import yolov5,torch
from yolov5.utils.general import (check_img_size, non_max_suppression,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import cv2
from PIL import Image as im
# Create your views here.
def index(request):
    return render(request,'index.html')
print(torch.cuda.is_available())
#load model
model = yolov5.load('best300.pt')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
device = select_device('') # 0 for gpu, '' for cpu
# initialize deepsort
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort('osnet_x0_25',
                    device,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    )
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names

def stream_drone1():
    rtmp_addr = "rtmp://34.243.140.144/live_hls/drone1"
    #cap = cv2.VideoCapture(rtmp_addr)
    cap_open = False
    while not cap_open:
        cap = cv2.VideoCapture()
        cap.open(rtmp_addr)
        print(f'CAP OPENED {cap.isOpened()}')
        cap_open = cap.isOpened()
    model.conf = 0.008
    model.iou = 0.5
    #model.classes = [0,64,39]
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture image")
            #break

        results = model(frame, augment=True)
        # proccess
        annotator = Annotator(frame, line_width=2, pil=not ascii) 
        det = results.pred[0]
        if det is not None and len(det):   
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):

                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]

                    c = int(cls)  # integer class
                    label = f'{id} {names[c]} {conf:.2f}'
                    annotator.box_label(bboxes, label, color=colors(c, True))

        else:
            deepsort.increment_ages()

        im0 = annotator.result()    
        print(im0)
        image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')  

def stream_drone2():
    rtmp_addr = "rtmp://34.243.140.144/live_hls/drone2"
    #cap = cv2.VideoCapture(rtmp_addr)
    cap_open = False
    while not cap_open:
        cap = cv2.VideoCapture()
        cap.open(rtmp_addr)
        print(f'CAP OPENED {cap.isOpened()}')
        cap_open = cap.isOpened()
    model.conf = 0.008
    model.iou = 0.5
    #model.classes = [0,64,39]
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture image")
            #break

        results = model(frame, augment=True)
        # proccess
        annotator = Annotator(frame, line_width=2, pil=not ascii)
        det = results.pred[0]
        if det is not None and len(det):
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):

                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]

                    c = int(cls)  # integer class
                    label = f'{id} {names[c]} {conf:.2f}'
                    annotator.box_label(bboxes, label, color=colors(c, True))

        else:
            deepsort.increment_ages()

        im0 = annotator.result()
        print(im0)
        image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')

def stream_drone3():
    rtmp_addr = "rtmp://34.243.140.144/live_hls/drone3"
    #cap = cv2.VideoCapture(rtmp_addr)
    cap_open = False
    while not cap_open:
        cap = cv2.VideoCapture()
        cap.open(rtmp_addr)
        print(f'CAP OPENED {cap.isOpened()}')
        cap_open = cap.isOpened()
    model.conf = 0.008
    model.iou = 0.5
    #model.classes = [0,64,39]
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture image")
            #break

        results = model(frame, augment=True)
        # proccess
        annotator = Annotator(frame, line_width=2, pil=not ascii)
        det = results.pred[0]
        if det is not None and len(det):
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):

                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]

                    c = int(cls)  # integer class
                    label = f'{id} {names[c]} {conf:.2f}'
                    annotator.box_label(bboxes, label, color=colors(c, True))

        else:
            deepsort.increment_ages()

        im0 = annotator.result()
        print(im0)
        image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')

def stream_camera1():
    rtmp_addr = "rtmp://34.243.140.144/live_hls/camera1"
    #cap = cv2.VideoCapture(rtmp_addr)
    cap_open = False
    while not cap_open:
        cap = cv2.VideoCapture()
        cap.open(rtmp_addr)
        print(f'CAP OPENED {cap.isOpened()}')
        cap_open = cap.isOpened()
    model.conf = 0.008
    model.iou = 0.5
    #model.classes = [0,64,39]
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture image")
            #break

        results = model(frame, augment=True)
        # proccess
        annotator = Annotator(frame, line_width=2, pil=not ascii)
        det = results.pred[0]
        if det is not None and len(det):
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):

                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]

                    c = int(cls)  # integer class
                    label = f'{id} {names[c]} {conf:.2f}'
                    annotator.box_label(bboxes, label, color=colors(c, True))

        else:
            deepsort.increment_ages()

        im0 = annotator.result()
        print(im0)
        image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')


def video_feed_drone1(request):
    return StreamingHttpResponse(stream_drone1(), content_type='multipart/x-mixed-replace; boundary=frame')
def video_feed_drone2(request):
    return StreamingHttpResponse(stream_drone2(), content_type='multipart/x-mixed-replace; boundary=frame')
def video_feed_drone3(request):
    return StreamingHttpResponse(stream_drone3(), content_type='multipart/x-mixed-replace; boundary=frame')
def video_feed_camera1(request):
    return StreamingHttpResponse(stream_camera1(), content_type='multipart/x-mixed-replace; boundary=frame')    

    

