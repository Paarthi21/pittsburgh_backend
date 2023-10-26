
#by rk

from fastapi import APIRouter,Request,HTTPException,status

from fastapi.responses import StreamingResponse

import pytesseract

from datetime import datetime

from creds import es
import usables

import json
import io

router = APIRouter(
    prefix='/elastic'
)


@router.post('/upload')
async def upload(request: Request):
    data = await request.json()

    # print(data)


    if not (data.get('filename',False) and data.get('size',False) and data.get('status',False) and data.get('dataurl',False) and data.get('username',False)):
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="provide both field and order key")

    username = data['username']
    
    if not usables.IsValidUser(username):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    
    indexname = usables.getIndex(username)
    status = data['status']
    res = []
    for fname,size,url in zip(data['filename'],data['size'],data['dataurl']):
        # print(fname)

        context = usables.data_url_to_image(url,fname)
        format = {
            "username":username,
            "filename":fname,
            "size":size,
            "context":context,
            "datetime":usables.getdatetime(),
            "dataurl":url,
            "status":status
        }

        
        fraw = usables.dataurltobytes(url)
        usables.UploadFileHdfs(file = fraw,filename = fname)

        res.append(es.index(index=indexname,document=format))
        # print(res[-1])

    return {"result":res}


@router.post('/search')

async def search(request: Request):
    data = await request.json()

    
    if (data.get('query',None) == None) or not (data.get('username',False)):
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="provide query key")

    
    username = data['username']
    if not usables.IsValidUser(username):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    
    
    indexname = usables.getIndex(username)
    # print(data['query'])
    if not data['query']:
        return es.search(index=indexname,query={"match_all":{}})['hits']['hits']

    query = {
        "query_string":{
        "query":f"*{data['query']}*",
        "fields":["filename","context"]
    }
    }
    result = es.search(index=indexname,query=query)['hits']['hits']

    return result

@router.post('/getallfiles')

async def getall(request: Request):

    data = await request.json()

    
    if not (data.get('username',False)):
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="provide username key")

    
    username = data['username']
    if not usables.IsValidUser(username):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    
    
    indexname = usables.getIndex(username)
    res = es.search(index=indexname,query={"match_all":{}})['hits']['hits']
    

    # async def filestream(res):
    #     for i in res:
    #         js = {"filename":i["_source"]['filename'],"dataurl":i["_source"]['dataurl']}
    #         yield json.dumps(js).encode("utf-8")+b"\n"


    # # response = [ for i in res if i]

    # return StreamingResponse(filestream(res), media_type="application/json")

    return res


@router.post('/sort')
async def sort(request:Request):
    data = await request.json()

    print(data)

    if not (data.get('name',False) and data.get('ord',False) and data.get('username',False)):
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="provide field,order,username key")

    
    username = data['username']
    if not usables.IsValidUser(username):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    
    
    indexname = usables.getIndex(username)


    query={f"{data['name']}":{"order":{data['ord']}}}

    return es.search(index=indexname,sort=query)['hits']['hits'] 



@router.post('/delete')
async def delete(request:Request):
    data = await request.json()

    if not (data.get('filename',False) and data.get('username',False)):
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="provide filename,username key")

    
    username = data['username']
    if not usables.IsValidUser(username):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    
    
    indexname = usables.getIndex(username)


    query = {
    "match_phrase":{
    "filename":data['filename']
    }
    }

    size = es.search(index="emp001",query=query)['hits']['hits']['_source']['size']
    size_mb = usables.size_converter(size,'bytes','MB')

    response = usables.update_size(username,size_mb,"-1")

    if response.get('status') == "exceeds":
        return json.dumps(response)

    return es.delete_by_query(index=indexname,query=query)


@router.post('/streamupload')
async def stream(request: Request):
    data = await request.json()

    # print(data)


    if not (data.get('filename',False) and data.get('size',False) and data.get('status',False) and data.get('dataurl',False) and data.get('username',False)):
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="provide both field and order key")

    username = data['username']
    
    if not usables.IsValidUser(username):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    
    indexname = usables.getIndex(username)
    status = data['status']

    def streaming_data():
        for fname,size,url in zip(data['filename'],data['size'],data['dataurl']):
            # print(fname)

            context = usables.data_url_to_image(url,fname)
            format = {
                "username":username,
                "filename":fname,
                "size":size,
                "context":context,
                "datetime":usables.getdatetime(),
                "dataurl":url,
                "status":status
            }

            print(es.index(index=indexname,document=format))
            
            size_mb = usables.size_converter(size,'bytes','MB')

            response = usables.update_size(username,size_mb,"1")

            if response.get('status') == "exceeds":
                return json.dumps(response)


            #you can do this using celery also (it is effective)
            # del format['datetime']
            fraw = usables.dataurltobytes(url)
            usables.UploadFileHdfs(file = fraw,filename = fname)

            yield json.dumps({'filename':format['filename']})
            # print(res[-1])

    return StreamingResponse(streaming_data(),media_type='application/json')





    
