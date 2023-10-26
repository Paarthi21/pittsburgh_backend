import pymongo
from fastapi import FastAPI,File,UploadFile, Form,Request,Depends,Body
from passlib.context import CryptContext
from jose import jwt
from fastapi import APIRouter
from token_authentication import token_required,SECRET_KEY
from passlib.context import CryptContext
import datetime
import creds
import usables


userrouter = APIRouter()


password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



@userrouter.post('/login')
def login(uid:str = Form(),password:str = Form()):
    collection = creds.database['users']
    flag = False
    name = ''
    role = ''
    uid1 = ''
    for post in collection.find():
        #print(uid,password)
        if (post['uid'] == uid and password_context.verify(password, post['password'])):
            flag = True
            name = post['name']
            uid1 = post['uid']
            role = post['role']
            dept = post['dept']
            break
    if flag == True:
        token = jwt.encode(
            {'uid': uid1, 'name': name, 'role': role,"dept":dept,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)},
            SECRET_KEY, "HS256")
        return {'token': token, "message": "Success"}

    else:
        return {"message": "Failure"}

@userrouter.post('/register',dependencies=[Depends(token_required)])
def data(name:str = Form(),password:str = Form(),uid:str = Form(),role:str = Form(),dept:str = Form()):
    collection = creds.database['users']
    data = {}
    myquery = {"uid": uid}
    data['name'] = name
    data['password'] = password_context.hash(password)
    data['uid'] = uid
    data['role'] = role
    data['dept'] = dept
    collection1 = creds.database['profile']
    collection1.delete_one(myquery)
    collection.insert_one(data)
    return "Success"

@userrouter.post('/deleteuser')
def data111(uid:str = Form()):
    collection = creds.database['users']
    data = {}
    myquery = {"uid": uid}
    collection.delete_one(myquery)
    collection1 = creds.database['profile']
    collection1.delete_one(myquery)
    return "Success"

@userrouter.post('/request',dependencies=[Depends(token_required)])
def data2(l1:str= Form(),l2:str= Form(),l3:str= Form(),dept:str = Form(),submitted:str = Form(),id:str = Form(),name:str = Form(),uid:str = Form(),role:str = Form(),status:str = Form(),data:str = Form()):
    collection = creds.database['request']
    data1 = {}
    data1['id'] = id
    data1['name'] = name
    data1['uid'] = uid
    data1['role'] = role
    data1['status'] = status
    data1['data'] = data
    data1['l1'] = l1
    data1['l2'] = l2
    data1['l3'] = l3
    data1['dept'] = dept
    data1['submitted'] = submitted
    collection.insert_one(data1)
    return "Success"
    

@userrouter.post('/requests',dependencies=[Depends(token_required)])
def data3(uid:str = Form()):
    collection1 = creds.database['request']
    lst = []
    uid = uid
    #print(uid)
    for x in collection1.find({"$and": [{"uid": {"$eq": uid}}]},
                                {"_id": 0, "id": 1, "name": 1, "uid": 1, "role": 1, "status": 1,"submitted": 1, "data": 1,"l1":1,"l2":1,"l3":1,"dept":1}):
        lst.append(x)
    return lst


@userrouter.post('/allrequests',dependencies=[Depends(token_required)])
def data4(token:str = Form()):
    collection1 = creds.database['request']
    lst = []
    for x in collection1.find({"$and": [{}]},
                                {"_id": 0, "id": 1, "name": 1, "uid": 1, "role": 1, "status": 1, "submitted":1 , "data": 1, "l1": 1,
                                "l2": 1, "l3": 1,"dept":1}):
        lst.append(x)
    return lst

@userrouter.post('/update',dependencies=[Depends(token_required)])
def data5(data:str = Form(),dept:str = Form(),id:str = Form(),name:str = Form(),uid:str = Form(),role:str = Form(),status:str = Form(),l1:str = Form(),l2:str = Form(),l3:str = Form(),submitted:str = Form()):
    collection1 = creds.database['request']
    lst = []
    data1 = {}
    myquery = {"id": id}
    collection1.delete_one(myquery)
    data1 = {}
    data1['id'] = id
    data1['name'] = name
    data1['uid'] = uid
    data1['role'] = role
    data1['status'] = status
    data1['data'] = data
    data1['l1'] = l1
    data1['l2'] = l2
    data1['l3'] = l3
    data1['dept'] = dept
    data1['submitted'] = submitted
    collection1.insert_one(data1)
    return "Success"

@userrouter.post('/profile',dependencies=[Depends(token_required)])
def data6(fname:str = Form(),lname:str = Form(),uid:str = Form(),email:str = Form(),bday:str = Form(),gender:str = Form(),add1:str = Form(),add2:str = Form(),phno1:str = Form(),phno2:str = Form(),state:str = Form(),linkedin:str = Form(),img:str = Form(),country:str = Form()):
    collection1 = creds.database['profile']
    lst = []
    myquery = {"uid": uid}
    collection1.delete_one(myquery)
    data1 = {}
    data1['fname'] = fname
    data1['lname'] = lname
    data1['uid'] = uid
    data1['email'] = email
    data1['bday'] = bday
    data1['gender'] = gender
    data1['country'] = country
    data1['add1'] = add1
    data1['add2'] = add2
    data1['phno1'] =phno1
    data1['phno2'] = phno2
    data1['state'] = state
    data1['linkedin'] = linkedin
    data1['img'] = img

    collection1.insert_one(data1)
    return "Success"



@userrouter.post('/getprofile',dependencies=[Depends(token_required)])
def data10(uid:str = Form()):
    collection1 = creds.database['profile']
    lst = []
    uid = uid
    #print(uid)
    for x in collection1.find({"$and": [{"uid": {"$eq": uid}}]},
                                {"_id": 0, "uid": 1, "fname": 1, "lname": 1, "bday": 1, "email": 1, "gender": 1,
                                "country": 1, "add1": 1, "phno1": 1, "phno2": 1, "add2": 1, "state": 1, "linkedin": 1, "img": 1}):
        lst.append(x)
    return lst



@userrouter.post('/getallemp',dependencies=[Depends(token_required)])
def data22(token:str = Form()):
    collection1 = creds.database['users']
    lst = []
    for x in collection1.find({"$and": [{}]},
                                {"_id": 0, "uid": 1, "name": 1, "role": 1, "dept": 1}):
        lst.append(x)
    return lst

@userrouter.post('/getalldept',dependencies=[Depends(token_required)])
def data23(token:str = Form()):
    collection1 = creds.database['departments']
    lst = list(collection1.find({"$and": [{}]},
                                {"_id": 0, "dept": 1}))
    #print(lst)

    return lst

@userrouter.post('/adddept',dependencies=[Depends(token_required)])
def data28(token:str = Form()):
    collection1 = creds.database['departments']
    lst = []
    # myquery = {"id": request.form.get("id")}
    # collection1.delete_one(myquery)
    data1 = {}
    data1['dept'] = ["digiverz","analytics","Human Resources"]
    collection1.insert_one(data1)
    return "Success"

@userrouter.post('/deleteemp',dependencies=[Depends(token_required)])
def data29(uid:str = Form()):
    collection = creds.database['users']
    data = {}
    myquery = {"uid": uid}
    collection.delete_one(myquery)
    collection1 = creds.database['profile']
    collection1.delete_one(myquery)
    return "Success"



@userrouter.post('/makeemp',dependencies=[Depends(token_required)])
def data534(uid:str = Form(),nuid:str = Form()):
    collection = creds.database['users']
    collection1 = creds.database['request']
    collection3 = creds.database['profile']
    uid = uid
    myquery = {"uid": uid}
    newvalues = {"$set": {"uid": nuid,"role":"Employee"}}
    collection.update_one(myquery, newvalues)
    myquery = {"uid": uid}
    newvalues = {"$set": {"uid": nuid, "role": "Employee"}}
    collection1.update_many(myquery, newvalues)
    newvalues = {"$set": {"uid": nuid}}
    collection3.update_many(myquery, newvalues)
    return "Success"

@userrouter.post('/regrade',dependencies=[Depends(token_required)])
def data535(uid:str = Form(),nuid:str = Form(),role:str = Form(),name:str = Form(),dept:str = Form()):
    collection = creds.database['users']
    collection1 = creds.database['request']
    collection3 = creds.database['profile']
    uid = uid
    myquery = {"uid": uid}
    newvalues = {"$set": {"uid": nuid, "role": role,"name": name,"dept": dept}}
    collection.update_one(myquery, newvalues)
    myquery = {"uid": uid}
    newvalues = {"$set": {"uid": nuid, "role": role,"name": name,"dept": dept}}
    collection1.update_many(myquery, newvalues)
    newvalues = {"$set": {"uid":nuid}}
    collection3.update_many(myquery, newvalues)
    return "Success"

@userrouter.post('/totalemp',dependencies=[Depends(token_required)])
def data5378(token:str = Form()):
    collection = creds.database['total']
    lst = list(collection.find({"$and": [{"id": {"$eq": 1}}]},
                                {"_id": 0, "total": 1}))

    #print(lst)
    return lst

@userrouter.post('/addtotal',dependencies=[Depends(token_required)])
def data5359(total:str = Form()):
    collection = creds.database['total']
    myquery = {"id": 1}
    newvalues = {"$set": {"total": total}}
    collection.update_one(myquery, newvalues)
    return "Success"

@userrouter.post('/totalreq',dependencies=[Depends(token_required)])
def data5379(token:str = Form()):
    collection = creds.database['total']
    lst = list(collection.find({"$and": [{"id": {"$eq": 2}}]},
                                {"_id": 0, "total": 1}))

    return lst

@userrouter.post('/addtotalreq',dependencies=[Depends(token_required)])
def data5389(total:str = Form()):
    collection = creds.database['total']
    myquery = {"id": 2}
    newvalues = {"$set": {"total": total}}
    collection.update_one(myquery, newvalues)
    return "Success"



@userrouter.post('/token',dependencies=[Depends(token_required)])
def tokencheck(token:str = Form()):
    return "Success"


@userrouter.post('/statusupdate',dependencies=[Depends(token_required)])
def update_from_search(uid,filename,status):
    collection = creds.database['request']
    lst = []
    uid = uid
    usables.status_update(filename=filename,uid=uid,status=status)
    collection.update_one({"uid":uid,"filename":filename},{ "$set": { "status": status } })
    return "Success"