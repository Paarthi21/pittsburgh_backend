import numpy as np
import io
import cv2
from token_authentication import token_required
from field_extraction import *
from PIL import Image
from fastapi import APIRouter
from token_authentication import token_required
from fastapi import FastAPI,File,UploadFile, Form,Request,Depends,Body
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# lm= WordNetLemmatizer()
# nltk.download('wordnet')
# stopwords.words("english")
# nltk.download('stopwords')
# nltk.download('punkt')


lang_input_glob = 'english'           

croprouter = APIRouter()
@croprouter.post('/crop')    
async def crop(request:Request):
    #print("inside crop") 
    fileslist = await request.json()
    image = fileslist['file_input']
    base_str = image
    if base_str is bytes:
        imageBinaryBytes = base_str
        image = np.asarray(bytearray(imageBinaryBytes), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        base_str = base_str.split(",")[1]
        binary = base64.b64decode(base_str)
        image = np.asarray(bytearray(binary), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # label = label_input
    #print("image",file_input)
    # image = io.BytesIO(image)
    # cropped_image = np.asarray(Image.open(image).convert('RGB'))
    # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
    #print("label at crop",label)
    # if label == 'from_address' or label=='to_address':
        #print("address")
        # #print(cropped_image.shape)
        # s = requests.post(url='https://ocr-tesseract-api2.herokuapp.com/predictocr',
        #                   json={"text_nlp[]": cropped_image.tolist(), "label": label})
    s = company_name_extract(image)
    text = s
    text = text.replace("\n"," ")
    return text

    # elif label == 'company_name':
    #     s = company_name_extract(cropped_image)
    #     text  = s
    #     #print("inside company",text)
    #     return text        
    # elif label == 'total':
    #     s = total(cropped_image,lang_input_glob)
    #     # s = requests.post(url='https://ocr-tesseract-api2.herokuapp.com/predictocr',
    #     #                   json={"text_nlp[]": cropped_image.tolist(), "label": label})
    #     # s = address(cropped_image)
    #     text = s
    #     #print(text)
    #     return text
    # elif label == 'invoice_number':
    #     # s = invoice_number(cropped_image)
    #     # s = requests.post(url='https://ocr-tesseract-api2.herokuapp.com/predictocr',
    #     #                   json={"text_nlp[]": cropped_image.tolist(), "label": label})
    #     # s = address(cropped_image)
    #     s = invoice_number(cropped_image,lang_input_glob)
    #     #print(s)
    #     text = s
    #     return text
    # elif label == 'invoice_date' or label == 'due_date':
    #     # s = date_extract(cropped_image)
    #     # s = requests.post(url='https://ocr-tesseract-api2.herokuapp.com/predictocr',
    #     #                   json={"text_nlp[]": cropped_image.tolist(), "label": label})
    #     # s = address(cropped_image)
    #     s = date_extract(cropped_image,lang_input_glob)
    #     text = s
    #     return text

    # elif label == 'phone_number':
    #     # s = date_extract(cropped_image)
    #     # s = requests.post(url='https://ocr-tesseract-api2.herokuapp.com/predictocr',
    #     #                   json={"text_nlp[]": cropped_image.tolist(), "label": label})
    #     # s = address(cropped_image)
    #     s = phone_number(cropped_image,lang_input_glob)
    #     text = s
    #     return text
    
    # elif label == 'sub_total':
    #     s =  total(cropped_image,lang_input_glob)
    #     text = s
    #     return text
    # elif label == 'logo':
    #     s = base64_create(cropped_image)
    #     return s
    # elif label == 'barcode':
    #      s = base64_create(cropped_image)
    #      return s
    # elif label == 'tax':
    #     s = tax_extract(cropped_image,lang_input_glob)
    #     return s
        
    # elif label == 'discount':
    #     s = discount_extract(cropped_image,lang_input_glob)
    #     return s
       

    # elif label == 'table':
    #     # s = requests.post(url='https://ocr-tesseract-api2.herokuapp.com/predictocr',
    #     #                   json={"text_nlp[]": cropped_image.tolist(), "label": label})
    #     s = table(cropped_image)
    #     headers_table = s[0]
    #     values_table = s[1]
    #     g = []
    #     for i in values_table:
    #         f = ''.join([j for j in " ".join(i) if not j.isdigit()])
    #         g.append(f)
    #     sentence = "".join(g)
    #     sentence = sentence.lower()
    #     words = nltk.word_tokenize(sentence.lower())
    #     new_words = [word for word in words if word.isalnum()]
    #     WordSet = []
    #     for word in new_words:
    #         if word not in set(stopwords.words("english")):
    #             WordSet.append(word)
    #     final_text = " ".join(WordSet)
    #     # category = categorize(final_text)
    #     # response_data = category.json()
    #     # category_data = response_data['category']
    #     # category_conf = response_data['category_conf']
    #     return {'headers':headers_table,'values':values_table}

    # else:
    #     s = address(cropped_image,lang_input_glob)
    #     #print(s)
    #     return s


