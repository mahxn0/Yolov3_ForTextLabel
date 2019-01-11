#!/usr/bin/env python
# -*- coding:utf-8 -*-
#import sys
import sys
#from PIL import Image
from PIL import Image,ImageDraw,ImageFont
import os
import time
import math
import numpy as np
import cv2
import sqlite3
#import queue as Queue
import Queue
import json

import sys
sys.path.append('/home/zdyd/workspace/src/detectAndRecog/src/')
import time, rospy, cv2, sys
import numpy as np
import thread, datetime, math,threading
import scipy.misc, scipy.io, os
#from pyzbar.pyzbar import decode

from sensor_msgs.msg import CompressedImage
from std_srvs.srv import *
from yidamsg.msg import LiveImage
#from yidamsg.msg import CaptureImage
from std_msgs.msg import String
from std_msgs.msg import Int64
from std_msgs.msg import Int16
#from sensor_msgs.msg import Image
from math import *
from yidamsg.msg import Log
from yidamsg.msg import InspectedResult
from yidamsg.msg import cam
from yidamsg.msg import result
#from yidamsg.msg import face

import base64
import urllib
import urllib2  #python2.7
import time
detect_recog = None

# text_sign import textRecognize_single
#from caffenet import caffenet_safety
from yolo_safely_helmet.src import yolo_safety
#from face_recog import facenet_Recog

class DetectAndRecognition(object):
    def __init__(self):
        self.yolo_detect = yolo_safety.yolo_safety()
        #self.text_recog = textRecognize_single.textrecognize()
        #self.safety_classify = caffenet_safety.caffenet()
        #self.faceRecog = facenet_Recog.FaceRecog()


    def recog_sign(self,img):
        if img is not None:
            sign_content = self.text_recog.recognize(img)
            return sign_content
        else:
            return False

    def face_recog(self,img):
        if img is not None:
            faceRecogRet = self.faceRecog.camera_recog(img)
            return faceRecogRet
        else:
            return False

def detectFailed(command_pub):
    msg = InspectedResult()
    msg.success = 0
    msg.camid = 1
    inspectedresult_pub.publish(msg)
    print('recognized failed')
    command_pub.publish('recognized')

def imageRotate(img, degree):

    height,width=img.shape[:2]
    heightNew=int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
    widthNew=int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))

    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)

    matRotation[0,2] +=(widthNew-width)/2
    matRotation[1,2] +=(heightNew-height)/2

    imgRotation=cv2.warpAffine(img,matRotation,(widthNew,heightNew),borderValue=(255,255,255))
    return imgRotation

def callback_camImage(data):
    print('received monitor pic')
    np_arr_front = np.fromstring(data.data, np.uint8)
    captureimg=cv2.imdecode(np_arr_front, cv2.IMREAD_COLOR)

    safetyData = {}
    safetyData['img'] = captureimg

    global safetyDataQueue
    safetyDataQueue.put(safetyData)

def faceRecog(data):
    print('received client data')
    np_arr_front = np.fromstring(data.facepic, np.uint8)
    captureimg=cv2.imdecode(np_arr_front, cv2.IMREAD_COLOR)
    camId = data.camid

    faceData = {}
    faceData['img'] = captureimg
    faceData['camId'] = camId

    global faceDataQueue
    faceDataQueue.put(faceData)

def changeModel(data):
    print('receive change model')
    print(type(data.data),data.data==1)
    if data.data == 1:
        global workModel
        workModel = 1

def sub_captureImage():
    rospy.Subscriber("/camera1/compressed",CompressedImage, callback_camImage,queue_size=20)
    rospy.spin()

def face_recognition():
    rospy.Subscriber('face_pic',face,faceRecog,queue_size=20)
    rospy.spin

def change_model():
    rospy.Subscriber('task_start', Int16, changeModel)
    rospy.spin

if __name__ == '__main__':
    global captureQueue
    captureQueue = Queue.Queue()
    rospy.init_node('fix_cam1', anonymous=True)
    detectResult_pub = rospy.Publisher("fix_cam1", cam, queue_size=1)
    signResult_pub = rospy.Publisher('sign_result',result,queue_size = 10)
    #faceResult_pub = rospy.Publisher('face_result',face,queue_size = 10)
    helmetPic_pub = rospy.Publisher('helmet_result',result,queue_size = 1)
    actionPic_pub = rospy.Publisher('aciton_result',result,queue_size = 1)

    thread.start_new_thread(sub_captureImage, ())
    #thread.start_new_thread(face_recognition,())
    thread.start_new_thread(change_model,())

    detect = DetectAndRecognition()
    global faceDataQueue
    faceDataQueue = Queue.Queue()

    global safetyDataQueue
    safetyDataQueue = Queue.Queue()

    rcvData = {}
    rcvData['img'] = cv2.imread('/home/zdyd/workspace/src/detectAndRecog/src/queue/test1.jpg')
    rcvData['camId'] = 1
    safetyDataQueue.put(rcvData)
    index = 0
    global workModel
    workModel = 1

    #workModel = 'safetySupervision'
    #'''
    #cap = cv2.VideoCapture('/home/zdyd/workspace/src/detectAndRecog/src/queue/video/3/out.avi')
    cap = cv2.VideoCapture()
    cap.open('rtsp://admin:123qweasd@192.168.8.98:554/h264/ch1/main/av_stream')

    #cap.set(3,1280)'rtsp://admin:asd123456@192.168.8.97'
    #cap.set(4,720)
    #cap.set(5,60)
    #cap.set(10,-4)
    #cap.set(11,40)
    #cap.set(12,20)
    #cap.set(15,-2)
    #'''
    while True:
        if True:
        #try:
            #print('workModel',workModel)
            t1 = time.time()
            global workModel
            if workModel == 0:
                global faceDataQueue
                if not faceDataQueue.empty():
                    data = faceDataQueue.get()
                    faceImg = data['img']
                    camId = data['camId']
                    ret = detect.face_recog(faceImg)
                    if len(ret) == 0:
                        msg = face()
                        msg.name = -1 #None
                        faceResult_pub.publish(msg)
                    else:
                        for res in ret:
                            msg = face()
                            msg.camid = camId
                            facepic = faceImg[res[0][1]:res[0][3],res[0][0]:res[0][2]]
                            msg.facepic = np.array(cv2.imencode('.jpg',facepic)[1]).tostring()
                            name = res[1]
                            score = res[2]
                            if name == 'yf' and int(score)>99:
                                msg.name = 1
                            elif name == 'lxr' and int(score)>99:
                                msg.name = 2
                            else:
                                msg.name = 0 #unkonw person
                            faceResult_pub.publish(msg)
                    print(ret)
            elif workModel == 1:
                global safetyDataQueue
                #if not safetyDataQueue.empty():
                #print('cap is opened',cap.isOpened())
                if cap.isOpened():
                #pathdir = '/home/zdyd/workspace/src/detectAndRecog/src/queue/video/1/pic/0'
                #filenames = os.listdir(pathdir)
                #for name in filenames:
                #safetyImg = cv2.imread(os.path.join(pathdir,name))

                #capRet,safetyImg = cap.read()
                #time.sleep(1)
                #if True:
                #if not safetyDataQueue.empty():
                    capRet,safetyImg = cap.read()
                    safetyImgDir = '/media/zdya/data'
                    t1 = time.time()
                    #data = safetyDataQueue.get()
                    #safetyImg = data['img']
                    #camId = data['camId']
                    imgName = '/home/zdyd/workspace/src/detectAndRecog/src/queue/test1.jpg'
                    #safetyImg = cv2.imread(imgName)
                    #cv2.imwrite(imgName,safetyImg)
                    print('xxxxxxxxxxxxxxxxxxxx')
                    out = detect.yolo_detect.detect(safetyImg)
                    t2=time.time()
                    #print(out)
                    msg = cam()
                    personNum = 0
                    helmetNum = 0
                    signImg = None
                    personImgList = []
                    personBoxList = []
                    helmetBoxList = []
                    actionBoxList = []
                    gloveBoxList = []
                    yandianBoxList = []
                    isGloveFlag = False
                    isYandianFlag = False
                    for ret in out:
                        detectClass = ret[0]
                        detectScore = float(ret[1])
                        left = int(ret[2])
                        top = int(ret[3])
                        right = int(ret[4])
                        bottom = int(ret[5])
                        validBox = False
                        if detectClass == 'person' and detectScore >= 0.4:
                            personNum += 1
                            personImg = safetyImg[top:bottom,left:right]
                            personImgList.append(personImg)
                            personBoxList.append([left,top,right,bottom])
	               		    cv2.putText(safetyImg, 'person', (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0),2)
			                cv2.rectangle(safetyImg,(left,top),(right,bottom),(255,0,0),2)
                        elif detectClass == 'safely_helmet' and detectScore >= 0.3:
                            helmetNum += 1
                            if top < 0:
                                top = 0
                            helmetImg = safetyImg[top:bottom,left:right]
                            personImgList.append(helmetImg)
                            cv2.rectangle(safetyImg,(left,top),(right,bottom),(0,255,0),2)
                            helmetBoxList.append([left,top,right,bottom])
                    '''
                    withoutHelmet = []
                    for personBox in personBoxList:
                        for helmetBox in helmetBoxList:
                            helmetCenterX = (helmetBox[1]+helmetBox[3])/2
                            helmetCenterY = (helmetBox[0]+helmetBox[2])/2
                            if helmetCenterX >= personBox[0] or helmetCenterX <= personBox[2] or \
                                helmetCenterY >= personBox[1] or helmetCenterY <= personBox[3]:
                                withoutHelmet.append(personBox)
                                continue
                    '''
                    if personNum > helmetNum:
                        msg.hat = 0
                    else:
                        msg.hat = 1
                    msg.campic = np.array(cv2.imencode('.jpg',safetyImg)[1]).tostring()
                    detectResult_pub.publish(msg)

                    print(t2-t1)
                    cv2.imshow("yolo",safetyImg)
                    cv2.waitKey(10)

