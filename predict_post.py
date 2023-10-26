import easyocr as ocr
from easyocr import Reader 
import cv2
import numpy as np
from ultralytics import YOLO
import imutils
import base64

predicted_response = {
    'first_name': '',
    'last_name': '',
    'address': '',
    'description': ''
    }

def empty_response():
    return {
    'first_name': '',
    'last_name': '',
    'address': '',
    'description': ''
    }



def predict(file_input):
    global predicted_response
    predicted_response = empty_response()
    base_str = file_input
    if base_str is bytes:
        imageBinaryBytes = base_str
        image = np.asarray(bytearray(imageBinaryBytes), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    elif type(file_input)==np.ndarray:
        image = file_input
    else:
        predicted_response['pdf_image'] = base_str
        base_str = base_str.split(",")[1]
        binary = base64.b64decode(base_str)
        image = np.asarray(bytearray(binary), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    exist_classes = []

    cap = image

    model = YOLO('postbest.pt')
    results = model(cap,conf=0.1)#show=True

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1) - 40, int(y1), int(x2) + 60, int(y2) + 50
            print(x1,y1,x2,y2)
            cv2.rectangle(cap,(x1,y1),(x2,y2),(255,0,255),3)
            #cv2.imshow('imge',cap)

            w,h = x2-x1,y2-y1
            cropped_image = cap[y1:y1 + h, x1:x1 + w]
            #cv2.imshow('imge',cropped_image)

            reader = ocr.Reader(['en'])
            result = reader.readtext(np.array(cropped_image))
            result_text = [] #empty list for results
            for text in result:
                result_text.append(text[1])

            name = result_text[0].split(' ')
            predicted_response['first_name'] = name[0]
            predicted_response['last_name'] = name[1]
            predicted_response['description'] =' '
            predicted_response['address'] = result_text[1:]
    print(predicted_response)
    return predicted_response