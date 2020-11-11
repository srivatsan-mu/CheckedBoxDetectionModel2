from __future__ import print_function
import requests
import json
global check
from PIL import  Image
from testtext import texti
from logosend import logosend
#import cv2
import convertapi
global dock
dock=list()
global fulltext
fulltext=dict()
global left,right,top,bottom
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

def closest(eps,K):
    min=300000000
    epsilon=tuple()
    for i in range(0,len(K)-1):
        X=(eps[0]-K[i][0])**2+(eps[1]-K[i][1])**2+(eps[2]-K[i][2])**2+(eps[3]-K[i][3])**2
        if(X<min):
            min=X
            epsilon=(K[i][0],K[i][1],K[i][2],K[i][3])
    print(epsilon)
    return epsilon



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
    logtires=logosend(img_file)
    img = open(img_file, 'rb').read()
    #immmg=cv2.imread(img_file,0)
    immg=Image.open(img_file)
    fulltext= texti(img_file)
    lateral_list=list(fulltext.keys())
    response = requests.post(test_url,img, headers=headers)
    #h,w=immmg.shape
    w,h=immg.size
    check=json.loads(response.text)
    lic=len(check['predictions'])
    x=0
    y=0
    xb=0
    yb=0
    for i in range(0,lic):
        if(check['predictions'][0]['probability']>0.2):
                dock.append(check['predictions'][0])
                lef.append(check['predictions'][0]['boundingBox']['left'])
                ltr=check['predictions'][0]['boundingBox']['left']
                x=(ltr)*w
                righ.append(check['predictions'][0]['boundingBox']['width'])
                rtr=check['predictions'][0]['boundingBox']['width']
                xb=(rtr)*w
                tp.append(check['predictions'][0]['boundingBox']['top'])
                ttr=check['predictions'][0]['boundingBox']['top']
                y=(ttr)*h
                botto.append(check['predictions'][0]['boundingBox']['height'])
                btr=check['predictions'][0]['boundingBox']['height']
                yb=(btr)*h
    print(fulltext)
    print(dock)
    eps=[0,0,0,0]
    epsom1=int(x)
    epsom2=int(y)
    epsom3=epsom1+int(xb)
    epsom4=epsom2+int(yb)
    eps=[epsom1,epsom2,epsom3,epsom4]
    print(eps)
    if (eps[0]==0 and eps[1]==0 and eps[2]==0 and eps[3]==0):
        return (logtires) 
    
    res9=closest(eps,lateral_list)
    #print(fulltext)
    thick=2
    color=(120,12,2)
    pt1=(epsom1,epsom2)
    pt2=(epsom3,epsom4)
    #out=cv2.rectangle(immmg,pt1,pt2,color,thick)
    #cv2.imwrite("pic.jpg",out)
    check="Form Sub Type :"+fulltext[res9]
    return (logtires,check)







