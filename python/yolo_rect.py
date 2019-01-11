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
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

# def c_array(ctype, values):
#     arr = (ctype*len(values))()
#     arr[:] = values
#     return arr

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
lib = CDLL("/home/mahxn0/darknet/darknet.so", RTLD_GLOBAL)
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
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int),c_int]
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

#net = load_net("/media/mahxn0/DATA/workspace/src/detectAndRecog/src/yolo_surface/data/robot/robot.cfg", "/media/mahxn0/DATA/workspace/src/detectAndRecog/src/yolo_surface/data/robot/robot_final.weights", 0)
#meta = load_meta("/media/mahxn0/DATA/workspace/src/detectAndRecog/src/yolo_surface/data/robot/robot.data")

net = load_net("/home/mahxn0/darknet/model/yolo_watch/watch.cfg", "/home/mahxn0/darknet/model/yolo_watch/watch.weights", 0)
meta = load_meta("/home/mahxn0/darknet/model/yolo_watch/watch.data")

#net = load_net("/home/mahxn0/ROS_workspace/darknet/cfg/yolov3-tiny.cfg", "/home/mahxn0/ROS_workspace/darknet/yolov3-tiny.weights", 0)
#meta = load_meta("/home/mahxn0/ROS_workspace/darknet/cfg/coco.data")
#video =cv2.VideoCapture(0)

class yolo_helmet(object):
    def __init__(self):
	    pass
    def detect_pic(self, image, thresh=0.6, hier_thresh=.5, nms=.45):
        im = self.nparray_to_image(image)
        num = c_int(0)
        pnum = pointer(num)
        predict_image(net, im)
        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms): do_nms_obj(dets, num, meta.classes, nms)
        res = []
        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    left=(b.x-b.w/2)
                    right=(b.x+b.w/2)
                    top=(b.y-b.h/2)
                    bot=(b.y+b.h/2)
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
        return res

    def detect(self, image, thresh=.3, hier_thresh=.5, nms=.45):
        t0=time.time()
        #rgbgr_image(im)
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
        print("detect take %d s",t2-t0);
        print("array_to_image take %d s",t1-t0)
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

    picDir = '/home/mahxn0/darknet/image/2018-11-29'
    print("path:",picDir)
    filenames = os.listdir(picDir)
    i=0
    font=cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    for name in filenames:
        filename = os.path.join(picDir,name)
        print(filename)
        image=cv2.imread(filename)

        r=yolo_helmet()
        out=r.detect(image)
        print(out)
        for res in out:
            x1=r.getXY(res[2])
            y1=r.getXY(res[3])
            x2=r.getXY(res[4])
            y2=r.getXY(res[5])
            frame_rect=image[y1:y2,x1:x2]
            cv2.imwrite('/home/mahxn0/darknet/image/watch_rect0.3/'+str(i)+'.jpg',frame_rect)
        i+=1



    # f_img=None
    # cap=cv2.VideoCapture()
    # cap.open("/media/mahxn0/Mahxn0/M_DataSets/jinan_data/Video/2018-07-07/192.168.0.222_01_20180707150311306.mp4")
#   #  cap.set(3,1280)
#   #  cap.set(4,720)
#   #  cap.set(5,60)
#   #  cap.set(10,-4)
#   #  cap.set(11,40)
#   #  cap.set(12,20)
#   #  cap.set(15,-2)
    # #cap.open('rtsp://admin:123qweasd@192.168.0.222:554/h264/ch1/main/av_stream')
    # cv2.namedWindow('YOLOV3')
    # r = yolo_helmet()
    # result = None
    # fileindex=0

    # font=cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    # #font = cv2.CAP_PVAPI_PIXELFORMAT_MONO8 # 使用默认字体
    # while(cap.isOpened()):
    # 	rect,frame=cap.read()
    #    frame_res=frame
    # 	if True:
    #         out = r.detect(frame)
 #  #         cv2.imshow("YOLOV3",frame)
    #         print(out)
    #         for res in out:
    #             x1=r.getXY(res[2])
    #             y1=r.getXY(res[3])
    #             x2=r.getXY(res[4])
    #             y2=r.getXY(res[5])
    #             frame_res=cv2.rectangle(frame, (x1,y1), (x2,y2), (87,255,123),4)
    #             cv2.putText(frame_res, res[0] + ' ' + str(res[1]), (x1,y1), font,1,(0,0,255),2)
    #             frame_rect=frame[x2:y2,x1:y1]
    #             cv2.imwrite("test.jpg",frame_rect)
    #    if  frame_res is None:
    #         print("frame_res is empty")
    #    else:
    #         cv2.imshow("YOLOV3",frame)
    #         cv2.waitKey(1)
