from __future__ import print_function
import requests
import json
global check
import math
import pandas as pd
from PIL import  Image
from testtext import texti
import pickle 
import convertapi
#import cv2
lef=[]
righ=[]
tp=[]
botto=[]
global textres
textres=list()
global logtitres
logtitres=list()
global clo
clo=0

knnreg=pickle.load(open('KNNRegressor.pkl','rb'))
knnclassifier=pickle.load(open('KNNClassifier.pkl','rb'))

def convertpdf(files):
    convertapi.api_secret = 'GOJ9sKouJI4lf9EY'
    convertapi.convert('jpg', {
        'File': files
    }, from_format = 'pdf').save_files('pdf2img/')

def sendi(pdf):
    files = "uploads/"+ pdf
    convertpdf(files)
    test_url = 'https://srivatsan.cognitiveservices.azure.com/customvision/v3.0/Prediction/09b6cec3-e166-44be-8830-b7c0804228d2/detect/iterations/Iteration20/image'
    headers = {'Prediction-Key': "015257f7093647ad9b57ac05e3b6c085","Content-Type":"application/octet-stream"}
    source=pdf[:-4]
    img_file="pdf2img/"+source+".jpg"
    img = open(img_file, 'rb').read()
    #immmg=cv2.imread(img_file)
    immg=Image.open(img_file)
    fulltext= texti(img_file)
    lateral_list=list(fulltext.keys())
    response = requests.post(test_url,img, headers=headers)
    #h,w=immmg.shape
    w,h=immg.size
    check=json.loads(response.text)
    lic=len(check['predictions'])
    j=1
    dock=list()
    for i in range(0,(lic-1)):
        if(check['predictions'][i]['probability']>0.10):
                # dock.append(check['predictions'][i])
                lef.append(check['predictions'][i]['boundingBox']['left'])
                ltr=check['predictions'][i]['boundingBox']['left']
                x=(ltr)*w
                righ.append(check['predictions'][i]['boundingBox']['width'])
                rtr=check['predictions'][i]['boundingBox']['width']
                xb=(rtr)*w
                tp.append(check['predictions'][i]['boundingBox']['top'])
                ttr=check['predictions'][i]['boundingBox']['top']
                y=(ttr)*h
                botto.append(check['predictions'][i]['boundingBox']['height'])
                btr=check['predictions'][i]['boundingBox']['height']
                yb=(btr)*h
                eps1=int(x)
                eps2=int(y)
                eps3=eps1+int(xb)
                eps4=eps2+int(yb)
                pim=[eps1,eps2,eps3,eps4]
                dock.append(pim)
    texter=list()
    for i in range (0,len(dock)-1):
        rar=[dock[i]]
        yll=knnreg.predict(rar)
        textstring=knnclassifier.predict(yll)
        texter.append(textstring)
    return texter









