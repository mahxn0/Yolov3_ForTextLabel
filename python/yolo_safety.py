#-*- coding=utf-8 -*-
from ctypes import *
import math
import random
import time
import cv2
import numpy as np
import re
import os
import sys
#import yolo_person
recordpath='/media/zdyd/code/yuanfei/jinan/yolo_safely_helmet/src/record/'
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    return (ctype * len(values))(*values)

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/media/zdyd/code/yuanfei/object_detection/darknet_industry/darknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

#net = load_net("/media/zdyd/code/yuanfei/jinan/yolo_safely_helmet/data/safely_helmet/yolov3-tiny.cfg", "/media/zdyd/code/yuanfei/jinan/yolo_safely_helmet/data/safely_helmet/yolov3-tiny_final.weights", 0)
net = load_net("/home/zdyd/workspace/src/detectAndRecog/src/yolo_safely_helmet/data/yolo_act/yolov3.cfg", "/home/zdyd/workspace/src/detectAndRecog/src/yolo_safely_helmet/data/yolo_act/yolov3_final.weights", 0)
meta = load_meta("/home/zdyd/workspace/src/detectAndRecog/src/yolo_safely_helmet/data/yolo_act/act.data")
video ='rtsp://admin:123qweasd@192.168.0.222:554/h264/ch1/main/av_stream'

class yolo_safety(object):
    def __init__(self):
	pass
    def detect(self, image, thresh=.1, hier_thresh=.5, nms=.45):
        #print(('array_to_image time: {}').format(time.time() - t))
    	#im=self.array_to_image(image)
        #im = load_image(image,0,0)
        #rgbgr_image(im)
        t0 = time.time()
        im = self.nparray_to_image(image)

        t1=time.time()
        num = c_int(0)
        pnum = pointer(num)
        predict_image(net, im)
        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum,0)
        num = pnum[0]
        if (nms): do_nms_obj(dets, num, meta.classes, nms)
        res = []
        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
		    left=b.x-b.w/2
                    right=b.x+b.w/2
                    top=b.y-b.h/2
                    bot=b.y+b.h/2
                    if left < 0:
                         left = 0
                    if right > im.w-1:
                         right = im.w-1
                    if top < 0:
                         top = 0
                    if bot > im.h-1:
                         bot = im.h-1
                    res.append((meta.names[i], dets[j].prob[i],left,top,right,bot))
        res = sorted(res, key=lambda x: -x[1])
        free_image(im) #not sure if this will cause a memory leak.
        free_detections(dets, num)
        t2=time.time()
        #print("detect take %d s",t2-t0);
        #print("array_to_image take %d s",t1-t0)
        return res
    def array_to_image(self,arr):
         arr = arr.transpose(2,0,1)
         c = arr.shape[0]
         h = arr.shape[1]
         w = arr.shape[2]
         arr = (arr/255.0).flatten()
         data = c_array(c_float, arr)
         im = IMAGE(w,h,c,data)
         return im

    def nparray_to_image(self,img):
        data = img.ctypes.data_as(POINTER(c_ubyte))
        image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
        return image

    def getXY(self,i):
        return int(i)
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    #图片检测
    # r = yolo_surface()
    # out = r.detect("/home/nvidia/workspace/src/detectAndRecog/src/yolo_surface/data/output.jpg")
    # print out
    #视频检测
    f_img=None
    cap=cv2.VideoCapture()
    cap.open(0)
    cap.set(3,1920)
    cap.set(4,1080)
    cap.set(5,60)
    cap.set(10,-4)
    cap.set(11,40)
    cap.set(12,20)
    cap.set(15,-2)
    #cap.open('rtsp://admin:123qweasd@192.168.0.222:554/h264/ch1/main/av_stream')
    cv2.namedWindow('YOLOV3')
    #r = yolo_person.yolo_person()
    r = yolo_safety()
    result = None
    fileindex=0
    while(cap.isOpened()):
    	rect,frame=cap.read()
        cv2.imwrite('./curImg.jpg',frame)
    	if True:
            t1=time.clock()
            #out = r.detect(frame)
            out = r.detect(frame)
            #t2 = time.clock()
            #print('detect time',t2-t1)
#           cv2.imshow("YOLOV3",frame)
            #print(out)
            frame_res=frame;
            for res in out:
                x1=r.getXY(res[2])
                y1=r.getXY(res[3])
                x2=r.getXY(res[4])
                y2=r.getXY(res[5])
                frame_res=cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255),4)
                y_limit=y1-100
                if y_limit<0:
                    y_limit=0
                frame_rect=frame[y_limit:y2,x1:x2]
                cv2.imwrite('1.jpg',frame_rect)
                #t1=time.clock()
                f_res=k.detect(frame_rect)
                #t2=time.clock()
                #print('helmet detect:',t2-t1)
                print('safely_helmet result %s',f_res)
                if len(f_res)!=0:
                    for fes in f_res:
                        x3=int(fes[2])
                        y3=int(fes[3])
                        x4=int(fes[4])
                        y4=int(fes[5])
                        x1result=x3+x1
                        y1result=y3+y_limit
                        x2result=x4+x1
                        y2result=y4+y_limit
                        frame_res=cv2.rectangle(frame_res,(x3+x1,y3+y_limit),(x4+x1,y4+y_limit),(0,255,0),4)
                       # shrink = cv2.resize(f_img, (640,480), interpolation=cv2.INTER_AREA)
                        fileindex+=1
            t2=time.clock()
            print('helmet detect:',t2-t1)
            cv2.imshow("YOLOV3",frame_res)
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
