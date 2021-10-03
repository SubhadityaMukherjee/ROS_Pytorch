#!/usr/bin/env python

#__________________________________
#|                                 |
#|           IMPORTS               |
#|_________________________________|

'''
#TODO
- change all the pooling to torch style
- add the rest of the models 
- add more models
'''

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
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

from PIL import Image

# from pytictoc import TicToc
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
# Network names might be a bit different
network_dict = {
    "resnet50": torchvision.models.resnet50(pretrained=True),
    "resnet34": torchvision.models.resnet34(pretrained=True),
    "densenet121": torchvision.models.densenet121(pretrained=True),
    "densenet161": torchvision.models.densenet161(pretrained=True),
    "densenet169": torchvision.models.densenet169(pretrained=True),
    "desnsenet201":torchvision.models.densenet201(pretrained=True),
    "googlenet":torchvision.models.googlenet(pretrained=True),
    "inception":torchvision.models.inception_v3(pretrained=True),
    "resnet101":torchvision.models.resnet101(pretrained=True),
    "resnet152":torchvision.models.resnet152(pretrained=True),
    "vgg16":torchvision.models.vgg16(pretrained=True),
}
if (base_network in network_dict.keys()):
    encoder = network_dict[base_network]
else:
    print("The selected network has not been implemented yet -- please choose another network!")
    exit() 

def feature_find(x_r,encoder):
    x_r = torch.unsqueeze(x_r, 0)
    encoder.eval()
    feature = encoder(x_r)
    feature = torch.squeeze(feature, 0)
    encoder.train()
    return feature


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

    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    ### deep feature vector of orthographic projections
    for i in range(0, number_of_views):
        
        img = all_images[i * number_of_bins * number_of_bins:(i + 1) * number_of_bins * number_of_bins]
        img = np.reshape(img, (number_of_bins, number_of_bins))
        max_pixel_value = np.max(img)
        img = img * (255 / max_pixel_value)

        resized_img, othographic_image = preprocessingForOrthographicImages(img, image_size)

        image_size = 224
        resized_img, othographic_image = preprocessingForOrthographicImages(img, image_size)

        img_g = cv2.merge((othographic_image, othographic_image, othographic_image))
        x_r = np.asarray(img_g)
        # x_r = np.expand_dims(x_r, axis=0)
        # print(x_r.shape)
        # exit()
        x_r = preprocess(x_r)
        feature = feature_find(x_r=x_r, encoder=encoder)
                
        # pooling functions
        feature = torch.tensor(feature)
        if (i == 0):
            global_object_representation = torch.tensor(feature)
        elif (pooling_function == "MAX"):
            global_object_representation = torch.max(torch.tensor(global_object_representation), feature)
        elif (pooling_function == "AVG"):
            global_object_representation = torch.average([global_object_representation, feature], axis=0)
        elif (pooling_function == "APP"):
            global_object_representation = torch.append(global_object_representation, feature, axis=1)
        


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
        # with graph.as_default():
        #     with session.as_default():    
        x_rgb = np.array(resized_rgb_img)
        # x_rgb = np.expand_dims(x_rgb, axis=0)
        # x_rgb = Image.fromarray(x_rgb)
        x_rgb = preprocess(x_rgb)
        feature = feature_find(x_r=x_rgb, encoder=encoder)
            
        # pooling functions # size of feature can be check first and then do this part
        if (pooling_function == "MAX"):
            # global_object_representation = torch.max([global_object_representation, feature], axis=0)

            global_object_representation = torch.max(torch.tensor(global_object_representation), feature)
        elif (pooling_function == "AVG"):
            global_object_representation = torch.average([global_object_representation, feature], axis=0)
        elif (pooling_function == "APP"):
            global_object_representation = torch.append(global_object_representation, feature, axis=1)

        #### encode Depth image
        x_depth = np.array(resized_depth_img)
        # x_depth = np.expand_dims(x_depth, axis=0)
        x_depth = preprocess(x_depth)
        feature = feature_find(x_r=x_depth, encoder=encoder)
            
        # pooling functions # size of feature can be check first and then do this part
        if (pooling_function == "MAX"):
            # global_object_representation = torch.max([global_object_representation, feature], axis=0)

            global_object_representation = torch.max(torch.tensor(global_object_representation), feature)
        elif (pooling_function == "AVG"):
            global_object_representation = torch.average([global_object_representation, feature], axis=0)
        elif (pooling_function == "APP"):
            global_object_representation = torch.append(global_object_representation, feature, axis=1)
        
    except CvBridgeError as e:
        print(e)
        print ("error visualize image")
    
    
    toc_global = time.clock()
    # print ("\t - size of representation is "+ str(global_object_representation.shape))
    # print ("\t - deep object representation took " + str (toc_global - tic_global))
    # print ("----------------------------------------------------------")            
    # print(np.array(global_object_representation)
    return deep_representationResponse(global_object_representation.detach().numpy())


def RGBD_multiview_service():
    rospy.init_node('deep_learning_representation_server')
    s = rospy.Service('RGBD_multiview_service', deep_representation, handle_deep_representation)
    print("Ready to representas RGBD object based on " + base_network + " network.")
    rospy.spin()

if __name__ == "__main__":
    RGBD_multiview_service()


