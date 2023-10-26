from fastapi import APIRouter, Form, HTTPException, Request,status


import usables
import creds


router = APIRouter(
    prefix='/stats'
)

@router.get('/')
def test():
    return "Welcome Stats"


@router.post('/DateBasedFrequency')

async def DateBasedFrequency(request: Request):

    
    
    data = await request.json()

    # print(data)


    if not data.get('username',False):
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="provide username key")

    username = data['username']
    
    if not usables.IsValidUser(username):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    response = []
    files_count_dates = {}
    files = usables.get_files(username)
    for record in files:
        date = record['_source']['datetime'].split("T")[0]
        if files_count_dates.get(date,False):
            files_count_dates[date]+=1
        else:
            files_count_dates[date]=1

    for Format,count in files_count_dates.items():
        response.append({"name":Format,"value":count})
    return response



@router.post('/TypeBasedFrequency')

async def TypeBasedFrequency(request: Request):
    
    data = await request.json()

    # print(data)


    if not data.get('username',False):
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="provide username key")

    username = data['username']
    
    if not usables.IsValidUser(username):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    
    
    response = []
    files_count_type = {}
    files = usables.get_files(username)
    for record in files:
        ext = record['_source']['filename'].split(".")[-1]
        if files_count_type.get(ext,False):
            files_count_type[ext]+=1
        else:
            files_count_type[ext]=1

    for Format,count in files_count_type.items():
        response.append({"name":Format,"value":count})
    return response




@router.post('/StorageDetails')

async def GetStorageUsed(request : Request):
    data = await request.json()

    # print(data)


    if not data.get('username',False):
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="provide username key")

    username = data['username']
    
    if not usables.IsValidUser(username):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    myquery = { "uid" : username}
    doc = creds.collection.find_one(myquery, {'_id': 0,"used_size":1,"total_size":1})

    return [{"storage":"Total Storage","value":doc['total_size'],"unit":"MB"},
            {"storage":"Used Storage","value":doc['used_size'],"unit":"MB"}]



@router.post('/GetActivity')

async def GetActivity(request: Request):
    data = await request.json()

    if not data.get('username',False):
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="provide username key")

    username = data['username']
    
    if not usables.IsValidUser(username):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    
    
    files = usables.get_files(username)

    res = [{"filename":record['_source']['filename'],"datetime":record['_source']['datetime']} for record in files]
        
    return res
