#!/usr/bin/env python

#__________________________________
#|                                 |
#|           IMPORTS               |
#|_________________________________|

### general
import roslib; #roslib.load_manifest('rug_deep_feature_extraction')
import rospy
from rug_deep_feature_extraction.srv import *
from nodelet.srv import *
import random
import os
import sys
import threading
import subprocess
import numpy as np
from tqdm import tqdm
import math
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from cv_bridge import CvBridge, CvBridgeError

import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

import torchvision
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

from PIL import Image

from pytictoc import TicToc
import time

# this is needed to get rid of cpu warning AUX
import os

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=np.inf)

def preprocessingForOrthographicImages (img, image_size):
    
    '''
        Note: since images are sparse, we need to apply dilation and erosion
    '''
    
    resized_img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img, dtype = np.uint8)

    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
    # img_resized_blur = cv2.blur(img_resized,(5,5))
    img_resized_blur = cv2.bilateralFilter(resized_img,5,15,15)
    # img_resized_blur = cv2.bilateralFilter(resized_img,5,75,75) modelNet

    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(img_resized_blur, kernel, iterations=1)
    othographic_image = 255 - dilation

    return resized_img, othographic_image


# "/vgg16_service", "/vgg19_service", "/xception_service", "/resnet50_service", "/mobilenet_service",  "/mobilenetV2_service", 
# "/densenet121_server", "/densenet169_server", "/densenet201_server", "/nasnet_large_server", "/nasnet_mobile_server", 
# "/inception_resnet_service", "/inception_service", "/autoencoder_service" 

base_network = str(sys.argv[1])
# print ("1 --- base_network = " + str(base_network))


recognition_network = "MobileNet"

# Load the Network.
    ### create the network model for object recognition part
if (base_network == "vgg16_fc1"):
    vgg_model = vgg16.VGG16(weights='imagenet', include_top=True)
    encoder = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc1').output)
    #vgg_model._make_predict_function()
    #plot_model(vgg_model, to_file='model.png')
    # print(vgg_model.summary())
else:
    print("The selected network has not been implemented yet -- please choose another network!")
    exit() 

              

tmp_img = 0

def handle_deep_representation(req):

    #__________________________
    #|                         |
    #|     ROS PARAMETERS      |
    #|_________________________|

    ## RULE: 0 : FALSE, 1 : TRUE
    image_normalization = 0 #FALSE 
    multiviews = 1 #TRUE 
    pooling_function = "MAX" #TRUE 
    number_of_bins = 150
    gui = True
    if rospy.has_param('/perception/image_normalization'):
        image_normalization = rospy.get_param("/perception/image_normalization")
        #print ("########## image_normalization (0 : FALSE, 1 : TRUE) = " + str(image_normalization))

    if rospy.has_param('/perception/multiviews'):
        multiviews = rospy.get_param("/perception/multiviews")
        #print ("########## multiviews (0 : FALSE, 1 : TRUE) = " + str(multiviews))

    if rospy.has_param('/perception/base_network'):
        base_network = rospy.get_param("/perception/base_network")
        # print ("\t - base_network = " + str(base_network))

    if rospy.has_param('/perception/pooling_function'):
        pooling_function = rospy.get_param("/perception/pooling_function")
        # print ("\t - pooling_function = " + str(pooling_function))
    
    if rospy.has_param('/perception/orthographic_image_resolution'):
        number_of_bins = rospy.get_param("/perception/orthographic_image_resolution")
        # print ("\t - orthographic_image_resolution = " + str(number_of_bins)) 

    if rospy.has_param('/perception/gui'):
        gui = rospy.get_param("/perception/gui")
        # print ("\t - gui = " + str(gui)) 

    #__________________________
    #|                         |
    #|    MAIN SERVICE CODE    |
    #|_________________________|

    number_of_views = int(len(req.good_representation) / (number_of_bins*number_of_bins))
    # print ("\t - number_of_views = " + str(number_of_views))
                
    image_size = 224
    #normalization 
    #The mean pixel values are taken from the VGG authors, 
    # which are the values computed from the training dataset.
    mean_pixel = [103.939, 116.779, 123.68]

    all_images = req.good_representation
    tic = time.clock()
    tic_global = time.clock()

    ### deep feature vector of orthographic projections
    for i in range(0, number_of_views):
        
        img = all_images[i * number_of_bins * number_of_bins:(i + 1) * number_of_bins * number_of_bins]
        img = np.reshape(img, (number_of_bins, number_of_bins))
        max_pixel_value = np.max(img)
        img = img * (255 / max_pixel_value)

        resized_img, othographic_image = preprocessingForOrthographicImages(img, image_size)

        with graph.as_default():
            with session.as_default():
                ## We represent each image as a feature vector
                ##TODO: image_size should be a param, some networks accept 300*300 input image
                image_size = 224
                resized_img, othographic_image = preprocessingForOrthographicImages(img, image_size)

                img_g = cv2.merge((othographic_image, othographic_image, othographic_image))
                x_r = image.img_to_array(img_g)
                x_r = np.expand_dims(x_r, axis=0)
                x_r = preprocess_input(x_r)
                feature = encoder.predict(x_r)          
                
        # pooling functions
        if (i == 0):
            global_object_representation = feature            
        elif (pooling_function == "MAX"):
            global_object_representation = np.max([global_object_representation, feature], axis=0)
        elif (pooling_function == "AVG"):
            global_object_representation = np.average([global_object_representation, feature], axis=0)
        elif (pooling_function == "APP"):
            global_object_representation = np.append(global_object_representation, feature, axis=1)


    ### deep feature vector of rgb and depth imgaes
    try:

        image_size = 224
        bridge = CvBridge()
        cv_rgb_image = bridge.imgmsg_to_cv2(req.RGB_image, "bgr8")
        resized_rgb_img, othographic_image = preprocessingForOrthographicImages(cv_rgb_image, image_size)                            

        if (gui):
            cv2.imshow('RGB_image', resized_rgb_img)
            cv2.waitKey(1)

        cv_depth_image = bridge.imgmsg_to_cv2(req.depth_image, "bgr8")      
        resized_depth_img, othographic_depth_image = preprocessingForOrthographicImages(cv_depth_image, image_size)                
    
        if (gui):
            cv2.imshow('Depth_image', resized_depth_img)
            cv2.waitKey(1)

        #### encode RGB image
        with graph.as_default():
            with session.as_default():    
                x_rgb = image.img_to_array(resized_rgb_img)
                x_rgb = np.expand_dims(x_rgb, axis=0)
                x_rgb = preprocess_input(x_rgb)
                feature = encoder.predict(x_rgb)
            
        # pooling functions # size of feature can be check first and then do this part
        if (pooling_function == "MAX"):
            global_object_representation = np.max([global_object_representation, feature], axis=0)
        elif (pooling_function == "AVG"):
            global_object_representation = np.average([global_object_representation, feature], axis=0)
        elif (pooling_function == "APP"):
            global_object_representation = np.append(global_object_representation, feature, axis=1)

        #### encode Depth image
        with graph.as_default():
            with session.as_default():          
                x_depth = image.img_to_array(resized_depth_img)
                x_depth = np.expand_dims(x_depth, axis=0)
                x_depth = preprocess_input(x_depth)
                feature = encoder.predict(x_depth)
            
        # pooling functions # size of feature can be check first and then do this part
        if (pooling_function == "MAX"):
            global_object_representation = np.max([global_object_representation, feature], axis=0)
        elif (pooling_function == "AVG"):
            global_object_representation = np.average([global_object_representation, feature], axis=0)
        elif (pooling_function == "APP"):
            global_object_representation = np.append(global_object_representation, feature, axis=1)

    except CvBridgeError as e:
        print(e)
        print ("error visualize image")
    
    
    toc_global = time.clock()
    # print ("\t - size of representation is "+ str(global_object_representation.shape))
    # print ("\t - deep object representation took " + str (toc_global - tic_global))
    # print ("----------------------------------------------------------")            
    return deep_representationResponse(global_object_representation[0])


def RGBD_multiview_service():
    rospy.init_node('deep_learning_representation_server')
    s = rospy.Service('RGBD_multiview_service', deep_representation, handle_deep_representation)
    print "Ready to representas RGBD object based on " + base_network + " network."
    rospy.spin()

if __name__ == "__main__":
    RGBD_multiview_service()


