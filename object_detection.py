# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 00:04:34 2019

@author: Arnav Kumar Nath
"""

import torch            #torch contains dynamic graphs to excute/compute gradients in backpropagation 
from torch.autograd import Variable #Module for gradient descent
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio     #Process the image of the video

#Defining a detect function
#net is the ssd neural network
#transformation of the image
#no need of gray image
def detect(frame, net, transform):
    height,width = frame.shape[:2]    #range from 0 height to 1width
    
    
    frame_t = transform(frame)[0]    #first element from two elements of the numpy array           
    x = torch.from_numpy(frame_t).permute(2,0,1)                             #transform into pytorch tensors from numpy arrays
                                                #red=0,blue=q,green=2
                                                #ssd only know grren,red,blue so we need to convert from rbg to grb
    #To add a fake dimension to create a btach as Neural network cannot accept single input but only batches of input
    x= Variable(x.unsqueeze(0))              #unsqueeze(0) to create a fake dimension and  index 0 is the 1st dimension corresponding to the batch
                                #to convert these batch of torchtensor of input into torch variable contains tensor and gradient and it becomes a element of dynamic graph which computes the graident effeciently during back propagation
                                
                                


#Input is ready to get into the network
    
    y=net(x)
        #To create a new tensor of 4 dimension
    detections = y.data                                
        #To normalise between 0 and 1 
    scale = torch.TensorClass([width,height,width,height])  #First two for upeer left and second two w,h for lower right
    
#Detections tensor contains 4 elements(batch; number of classes = objects that can be detected eg:dog,boats,cars; the number of occurence of the class ; tuple of 5 elements(score, x0,y0,x1,y1 for each occurence of each class we get these numbers ) )  
#score < 0.6 then occurence will  not be found and x                       
#Loop through all the classes if score < 0.6
    for i in range(detections.size(1)):
        j = 0  #j= Occurence while i = class
        while detections[0, i, j, 0] >= 0.6
            pt = (detections[0, i, j, 1:] * scale).numpy()
    #Putting the tensor back to numpy o that opencv can draw rectangle
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)        #pt[0]= x0 and pt[1]= upper right corner as one argument
            cv2.putText(frame, labelmap[i-1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,2), 2, cv2.LINE_AA )      #LINE_AA gives us continuoys while displaying
             j += 1
    return frame

#creating the SSD Neural Network by testing on a video
#load the  pre-trained network and load
net = build_ssd('test')
net.load_state_dic(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc : storage))


#Creating the transformation
#net.size is the target size of the image to be fitted and scale number means the weight under which neural network is trained
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) 




#Object Detection on a video
reader = imageio.get_reader('funny_dog.mp4')
fps    = reader.get_meta_data()['fps']
#Create a output video with the same fps and rectangles
writer = imageio.get_writer('output_funny_dog.mp4', fps = fps)

#Convert the frame into network
for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    
#Append the frame into output video
    writer.append_data(frame)
    print(i)         

writer.close()     