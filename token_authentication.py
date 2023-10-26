import jwt
from fastapi import FastAPI,File,UploadFile, Form,Request,Depends,Body

SECRET_KEY = '004f2af45d3a4e161a7dd2d17fdae47f'



def token_required(token:str = Body()):
    if not token:
        return {'message': 'a valid token is missing'}
    try:
        data6 = jwt.decode(token,SECRET_KEY, algorithms=["HS256"])
    except:
        return {'message': 'token is invalid'}