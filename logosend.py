from __future__ import print_function
import requests
import json
global logtitle
logtitle=list()
def logosend(img_file):
    test_url = 'https://srivatsan.cognitiveservices.azure.com/customvision/v3.0/Prediction/4356d87c-8852-4b9c-8a39-5c6f59a62acb/detect/iterations/Iteration8/image'
    #test_url = addr + '/api/test'
    # prepare headers for http request
    headers = {'Prediction-Key': "015257f7093647ad9b57ac05e3b6c085","Content-Type":"application/octet-stream"}
    #img_file = 'image0.png'
    # encode image as jpe
    # print(img_file)
    # images = convert_from_path(img_file)
    # for i, image in enumerate(images):
    #     fname = "static/image" + str(i) + ".png"
    #     image.save(fname, "PNG")
    #     break
    # img_file = 'static/image0.png'
    #_, img_encoded = cv2.imencode('.jpg', img)
    img = open(img_file, 'rb').read()

    # send http request with image and receive response
    response = requests.post(test_url,img, headers=headers)
    # decode response
    logtitle=[]
    print(json.loads(response.text)['predictions'][0])

    print(json.loads(response.text)['predictions'][1])

    if(json.loads(response.text)['predictions'][0]['probability'] > 0.35):
        logo=json.loads(response.text)['predictions'][0]['tagName']
        logtitle.append(logo)
    if(json.loads(response.text)['predictions'][1]['probability'] > 0.35):
        title=json.loads(response.text)['predictions'][1]['tagName']
        logtitle.append(title)
    return logtitle

