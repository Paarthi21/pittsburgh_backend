import uvicorn
from fastapi import FastAPI,File,UploadFile, Form,Request,Depends,Body
from typing import List
from pydantic import BaseModel
import pickle
import numpy as np
import argparse
import os
import cv2
from werkzeug.utils import secure_filename
from io import BytesIO
import glob
import requests
import re
from os.path import join, dirname, realpath
import base64
from PIL import Image, ImageFile  # from Pillow
from colorama import Fore  # from native modules
import platform  # from native modules
from os.path import join, dirname, realpath
import json
from detect import start
import io  # from native modules
import platform  # from native modules
import nltk
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from functools import wraps
import datetime
from base64 import b64encode,decodestring
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import dataclass, field
from preprocessing_image import deskew,erode,canny,remove_noise,get_grayscale,match_template,dilate,opening,thresholding
from preprocessing_text import seperate,correction
from field_extraction import *
from token_authentication import token_required
from nltk.corpus import stopwords
import user_management
from text_categorizing import categorize
import crop_endpoint
import routes.elastic_search as elastic
import pytesseract
import creds


# pytesseract.pytesseract.tesseract_cmd = 'C:/Users/lcharankumar/AppData/Local/Tesseract-OCR/tesseract.exe'
#pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
from fastapi_utils.tasks import repeat_every
import usables




lang_input_glob = 'english'           
predicted_response = {
    'company_name':'',
    'from_address':'',
    'to_address':'',
    'invoice_date':'',
    'due_date':'',
    'phone_number':'',
    'invoice_number':'',
    'currency':'',
    'total':'',
    'sub_total':'',
    'tax':'',
    'discount':'',
    'barcode':'',
    'logo':'',
    'custom': [],
    'category':'',
    'pdf_image':None,
    'bill_of_materials':[
        {
            "description":[],
            "quantity":[],
            "unit_price":[],
            "price":[],
        }
    ]
}

def empty_resp():
    return {
    'company_name':'',
    'from_address':'',
    'to_address':'',
    'invoice_date':'',
    'due_date':'',
    'phone_number':'',
    'invoice_number':'',
    'currency':'',
    'total':'',
    'sub_total':'',
    'tax':'',
    'discount':'',
    'barcode':'',
    'logo':'',
    'custom': [],
    'category':'',
    'pdf_image':None,
    'bill_of_materials':[
        {
            "description":[],
            "quantity":[],
            "unit_price":[],
            "price":[],
        }
    ]
}

pickled_model = pickle.load(open('textmodel2.pkl', 'rb'))


def predict(file_input,lang_input):
    global predicted_response
    predicted_response = empty_resp()
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
    #print("image",image)
    global lang_input_glob
    lang_input_glob = lang_input
    # choice = ['invono','date','total','table','address']
    exist_classes = []
    ##print(os.path.join(UPLOAD_FOLDER, filename))
    #image.save(os.path.join('/tmp',filename))]
    classes = ['company_name', 'from_address', 'to_address', 'date', 'phone_number', 'invoice_number', 'total', 'sub_total', 'tax', 'discount', 'barcode', 'logo', 'description_col', 'qty_col', 'price_col', 'unitprice_col', 'header', 'table']
    # read input image
    image2 = image
    image3 = image
    # image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
    # image3 = cv2.cvtColor(image3, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("sample.png",image2)
    #print("image2",image2)
    #print("model selected: ",creds.modelToLoad)
    values, img = start("best_1000.pt", image2)
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    dw = image2.shape[1]
    dh = image2.shape[0]
    table_values = []
    currency = ''
    header = []
    column_count = 0
    header_len = 0
    #print("label index:",[i[0] for i in values])
    for i in values:
        class_id = int(i[0])
        w = i[3]
        h = i[4]
        x = i[1]
        y = i[2]
        confidence = i[5]
        #print(class_id)
        label = str(classes[class_id])
        x_center, y_center, w, h = float(x), float(y), float(w), float(h)
        x_center = round(x_center * dw)
        y_center = round(y_center * dh)
        w = round(w * dw)
        h = round(h * dh)
        x = round(x_center - w / 2)
        y = round(y_center - h / 2)
        cropped_image = image3[y:y + h, x:x + w]
        # color = COLORS[int(class_id)]
        # #print("+",x+w, y+h)
        # cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        #cv2.imwrite(os.path.join(app.config['CROPPED_FOLDER'], str(classes[class_id]) + '.png'), cropped_image)
        #cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        #print("label->",label)
        if label == 'from_address':
            # s = address(cropped_image)
            predicted_response['from_address'] = address(cropped_image,lang_input)
        elif label == 'to_address':
            predicted_response['to_address'] = address(cropped_image,lang_input)
        elif label == 'total':
            predicted_response['total'] = total(cropped_image,lang_input)
            if currency=='':
               currency = currency_extract(cropped_image,lang_input)
               predicted_response['currency'] = currency
        elif label == 'invoice_number':
            predicted_response['invoice_number'] = invoice_number(cropped_image,lang_input)
        elif label == 'date':
            date_res = date_extract(cropped_image,lang_input)
            """ print("date_res",date_res) """
            if list(date_res.keys())[0]=="invoice_date":
               predicted_response['invoice_date'] = date_res['invoice_date']
            if list(date_res.keys())[0]=="due_date":
               predicted_response['due_date'] = date_res['due_date']
        elif label == 'company_name':
             predicted_response['company_name'] = company_name_extract(cropped_image)
        elif label == 'qty_col':
             if lang_input=='english':
                predicted_response['bill_of_materials'][0]['quantity'] = [] if col_extract(cropped_image)==None else col_extract(cropped_image)
                
             elif lang_input=='arabic':
                predicted_response['bill_of_materials'][0]['quantity'] = arabic_col_extract(cropped_image)
             column_count+=1
        elif label == 'price_col':
             if lang_input=='english':
                predicted_response['bill_of_materials'][0]['price'] = [] if col_extract(cropped_image)==None else col_extract(cropped_image)
                
             elif lang_input=='arabic':
                predicted_response['bill_of_materials'][0]['price'] = arabic_col_extract(cropped_image)
             column_count+=1         
        elif label == 'description_col':
             if lang_input=='english':
                predicted_response['bill_of_materials'][0]['description'] = [] if col_extract(cropped_image)==None else col_extract(cropped_image)
                
             elif lang_input=='arabic':
                predicted_response['bill_of_materials'][0]['description'] = arabic_col_extract(cropped_image)
             column_count+=1

        elif label == 'unitprice_col':
            if lang_input=='english':
                predicted_response['bill_of_materials'][0]['unit_price'] = [] if col_extract(cropped_image)==None else col_extract(cropped_image)
                
            elif lang_input=='arabic':
                predicted_response['bill_of_materials'][0]['unit_price'] = arabic_col_extract(cropped_image)
            column_count+=1
          
        elif label == 'sub_total':
            predicted_response['sub_total'] = total(cropped_image,lang_input)
            if currency=='':
               currency = currency_extract(cropped_image,lang_input)
               predicted_response['currency'] = currency
        elif label == "phone_number":
            predicted_response['phone_number'] = phone_number(cropped_image,lang_input)
        elif label == 'logo':
             predicted_response['logo'] = base64_create(cropped_image)
        elif label == 'barcode':
             predicted_response['barcode'] = base64_create(cropped_image)
        elif label == 'tax':
            predicted_response['tax'] = tax_extract(cropped_image,lang_input)
            if currency=='':
               currency = currency_extract(cropped_image,lang_input)
               predicted_response['currency'] = currency
        elif label == 'discount':
            predicted_response['discount'] = discount_extract(cropped_image,lang_input)
            if currency=='':
               currency = currency_extract(cropped_image,lang_input)
               predicted_response['currency'] = currency
        elif label == 'header':
             header = table(cropped_image)[0]
             #print("headers 1",header)
             header_len = len(header)
        elif label == 'table':
            table_img = cropped_image
            s = table(cropped_image)
            table_values = s[1]
            

    # if header_len!=column_count and column_count<4:
    #    table_values = table(table_img)[1]
    #    #print("values",table_values)
    #    #print("header",header)
    #    #print("column",column_count)

    # if header_len>column_count and column_count<4:
    #     for i,j in enumerate(predicted_response['bill_of_materials'][0]):
    #         if len(predicted_response['bill_of_materials'][0][j])==0:
    #             if len(table_values[0])==header_len:
    #                 #print(i)
    #                 predicted_response['bill_of_materials'][0][j]=[x[i] for x in table_values if len(x)>=i]
    #             elif len(table_values[0])>header_len:
    #                  table_values = [x[1::] for x in table_values]
    #                  predicted_response['bill_of_materials'][0][j]=[x[i] for x in table_values if len(x)>=i]



    
    g = []
    for i in table_values:
        f = ''.join([j for j in " ".join(i) if not j.isdigit()])
        g.append(f)
    sentence = "".join(g)
    sentence = sentence.lower()
    words = nltk.word_tokenize(sentence.lower())
    new_words = [word for word in words if word.isalnum()]

    WordSet = []
    for word in new_words:
        if word not in set(stopwords.words("english")):
            WordSet.append(word)
    final_text = " ".join(WordSet)
    predicted_response['category'] = categorize(final_text)[0]



    return predicted_response