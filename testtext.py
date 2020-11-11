from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time
#import cv2
global text
text=dict()
global sum
def texti(img_path):
    text=dict()
    endpoint="https://texti.cognitiveservices.azure.com/"
    subscription_key="855ccecb6ea649c1a3372d5758c4b4ed"
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    local_image_printed_text_path = img_path
    local_image_printed_text = open(local_image_printed_text_path, "rb")
    #img=cv2.imread(img_path,0)
    ocr_result_local = computervision_client.recognize_printed_text_in_stream(local_image_printed_text)
    for region in ocr_result_local.regions:
        for line in region.lines:
            #print("Bounding box: {}".format(line.bounding_box))
            s=''
            for word in line.words:
                s += word.text + " "
                sum=0
                #print(s)
                #print(line.bounding_box)
                L = [int(x) for x in line.bounding_box.split(',')] 
                #print(L)
                emp1= L[0]
                emp3=L[0]+L[2]
                emp2=L[1]
                emp4=L[1]+L[3]
                #pt1=(emp1,emp2)
                #pt2=(emp3,emp4)
                #thick=2
                #color=(120,56,90)

                #out=cv2.rectangle(img,pt1,pt2,color,thick)
                #cv2.imwrite("john.jpg",out)
                #print(emp1,emp2,emp3,emp4)
                # for i in range(0,len(L)):

                # # for i in range(0,(len(line.bounding_box)-1)):
                # #     if(line.bounding_box[i]!=','):
                # #         print(line.bounding_box[i])
                # #         sum=sum+int(line.bounding_box[i])
            #print(s)
            text[(emp1,emp2,emp3,emp4)]=s   
           # print(s)
            #hes=int(line.bounding_box[0:2])+int(line.bounding_box[3:5])+int(line.bounding_box[6:8])+int(line.bounding_box[9:10])
            #print(hes)
        
            #text[line.bounding_box]=s
            #print()
    return text
    
    