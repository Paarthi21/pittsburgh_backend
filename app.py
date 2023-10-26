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
from base64 import b64encode
# from fastapi.middleware.cors import CORSMiddleware
from dataclasses import dataclass, field
from preprocessing_image import deskew,erode,canny,remove_noise,get_grayscale,match_template,dilate,opening,thresholding
from preprocessing_text import seperate,correction
from field_extraction import *
from user_management import tokencheck
from token_authentication import token_required
from nltk.corpus import stopwords
import user_management
from text_categorizing import categorize
import crop_endpoint
import routes.elastic_search as elastic
import pytesseract
from predict_data import predict
import predict_data
from fastapi_utils.tasks import repeat_every
import usables
from usables import predict_from_mail
from starlette.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
import os
import routes.stats as stats
import hashlib
from difflib import SequenceMatcher
import creds
import predict_post

print(os.getenv('elastic_url'))

# pytesseract.pytesseract.tesseract_cmd = 'C:/Users/lcharankumar/AppData/Local/Tesseract-OCR/tesseract.exe'


app = FastAPI()


@app.on_event("startup")
@repeat_every(seconds=10) 
def tasks():
    predict_from_mail()



origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_management.userrouter)
app.include_router(crop_endpoint.croprouter)
app.include_router(elastic.router)
app.include_router(stats.router)

# poppler_path = r"D:/invoice-management/poppler-22.04.0/Library/bin"
#poppler_path = r"D:/projects/ocr-search/poppler-22.04.0/Library\bin"
#poppler_path= r"C:/Users/paart/Videos/poppler-23.07.0/Library/bin"


#converting the sample files into byte codes
testFiles=usables.pathtobytes(creds.testFilePath)
""" print("Array length", len(testFiles)) """


#/predict api
@app.post('/predict')
async def predicted_output(request:Request):
    
    
    fileslist = await request.json()
    fileslist = fileslist['data']
    #print(fileslist)
    res = []
    count = 0
    image_lists = []
    
 
    for i in fileslist:
        #print("count",count)
        ext = str(i['name']).split(".")[-1]

        
        if ext=="pdf":

            
            base64string = str(i['data']).split(",")[1]
            bytes_str = base64.b64decode(base64string)
            # images = convert_from_bytes(bytes_str,poppler_path=poppler_path,fmt="png")
            
            #####deciding which model to render###### 
            creds.modelToLoad=usables.file_compare(bytes_str,testFiles)
            """ print("selected model: ", creds.modelToLoad) """
            images = convert_from_bytes(bytes_str,fmt="png")
            for i in range(len(images)):
                decoded = np.array(images[i].convert('RGB'))
                image_lists.append(decoded)
                
            im_v = cv2.vconcat(image_lists)
            _, im_arr = cv2.imencode('.jpg', im_v)  # im_arr: image in Numpy one-dim array format.
            im_bytes = im_arr.tobytes()
            im_b64 = base64.b64encode(im_bytes)
            dataurl = f'data:image/jpg;base64,{str(im_b64)[2:len(im_b64)-2]}'
            # print(len(dataurl))
            output = predict_data.predict(dataurl,"english")
            res.append(output)
            count+=1

          
        else:  
            output = predict_post.predict(str(i['data']))
            res.append(output)
            count+=1  
            
    return {"response":res}

""" @app.get('/get')
async def test():
    print("Hello")
    return "api ok"  """
