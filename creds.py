#by rk
from elasticsearch import Elasticsearch
import imaplib
from dotenv import load_dotenv
import os
import email
from pymongo import MongoClient
from pywebhdfs.webhdfs import PyWebHdfsClient

load_dotenv('./env')



# es = Elasticsearch(["https://localhost:9200"],basic_auth=("elastic","_y=PZ4+SNmrs2Tq6m7CV"),verify_certs=False)

#mongo db
uri = os.getenv('mongo_uri')
client = MongoClient(uri)
database = client['invoice']
collection = database['users']

elastic_collection = database['elastic_creds']
gmail_collection = database['gmail_creds']
hdfs_collection = database['hdfs_creds']


elastic_creds = elastic_collection.find_one({},{"_id":0})
hdfs_creds = hdfs_collection.find_one({},{"_id":0})
gmail_creds = gmail_collection.find_one({},{"_id":0})



es = Elasticsearch([elastic_creds['elastic_url']],basic_auth=(elastic_creds['elastic_username'],elastic_creds['elastic_password']),verify_certs=False)


modelToLoad='best_1000.pt'

mail = imaplib.IMAP4_SSL("imap.gmail.com",993)

testFilePath=r"./Sample_files"

mail.login(gmail_creds['email'],gmail_creds['mail_pass'])

poppler_path="C:\applications\Release-23.07.0-0\poppler-23.07.0\Library\bin"

mail.select('Inbox')

hdfs= PyWebHdfsClient(host=hdfs_creds['hdfshost'],port=hdfs_creds['hdfsport'],user_name=hdfs_creds['hdfsuser'])