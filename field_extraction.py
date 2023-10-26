from preprocessing_image import deskew,erode,canny,remove_noise,get_grayscale,match_template,dilate,opening,thresholding
import cv2
import numpy as np
from preprocessing_text import seperate,correction
import pytesseract
from pytesseract import Output
import pandas as pd
import re
import base64

# pytesseract.pytesseract.tesseract_cmd = 'C:/Users/lcharankumar/AppData/Local/Tesseract-OCR/tesseract.exe'

def total(img,lang_input):
    # Adding custom options
    if lang_input=='english':
        custom_config = r'-l eng --oem 3 --psm 11 -c tessedit_char_blacklist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$£€¥" -c tessedit_char_whitelist="0123456789,."'
        canny = get_grayscale(img)
        h, w = canny.shape
        d = pytesseract.image_to_data(canny, config=custom_config, output_type=Output.DICT)
        # d = correction(d['text'])
        d = "".join(d['text'])
        d = re.findall(r'[\d,.$£€¥₹]*', d)
        d = "".join(d)
        return d

    elif lang_input == 'arabic':
        custom_config = r"--psm 11 --oem 3 -c tessedit_char_whitelist=1234567890,."
        img = get_grayscale(img)
        result = pytesseract.image_to_string(img, lang='ara',config=custom_config)
        return [i for i in result.split("\n") if i] if len([i for i in result.split("\n") if i])>1 else result.strip("\n .")

        

def address(img,lang_input):
    # Adding custom options
    if lang_input == 'english':
        custom_config = r'-l eng --psm 11 --oem 3'
        canny = get_grayscale(img)
        ignore_words = [ 'SHIPTO', 'BILLEDTO', 'YOURCOMPANYNAME', 'FROM', 'TO', 'SHIP', 'SHIPPING', 'BILLOF', 'ADDRESS', 'BILLING', 'BILLTO', 'SHIPTO', 'SHIPPINGTO', 'BILLINGTO','SHIP TO', 'BILLED TO', 'YOUR COMPANY NAME', 'FROM', 'TO', 'SHIP', 'SHIPPING', 'BILL OF', 'ADDRESS', 'BILLING', 'BILL TO', 'SHIP TO', 'SHIPPING TO', 'BILLING TO',"ship to","billed to",'your company name','from','to','ship','shipping','bill of','address','billing','bill to','ship to','shipping to','billing to','shipto', 'billedto', 'yourcompanyname', 'from', 'to', 'ship', 'shipping', 'billof', 'address', 'billing', 'billto', 'shipto', 'shippingto', 'billingto']
        d = pytesseract.image_to_string(canny, config=custom_config,output_type=Output.DICT)
        #print(d['text'])
        d = d['text'].replace('\n',' ')
        d = d.split(' ')
        d = [x for x in d if x.lower() not in ignore_words]
        d = " ".join(d).replace('  ','\n')
        return d
    elif lang_input == 'arabic':
        img = get_grayscale(img)
        custom_config = r'-l ara --oem 3 --psm 6'
        d = pytesseract.image_to_string(img, config=custom_config)
        return d

    


def invoice_number(img,lang_input):
    if lang_input == 'english':
        custom_config = r'-l eng --psm 11 --oem 3'
        canny = get_grayscale(img)
        d = pytesseract.image_to_data(canny, config=custom_config, output_type=Output.DICT)
        # d = correction(d['text'])
        #print(d['text'])
        d = seperate(d['text'])
        #print("after sep",d)
        try:
           return " ".join(d[1:]) if len(d)>1 else d[0].split(" ")[-1]
        except:
            pass
    elif lang_input=='arabic':
        custom_config = r'-l ara+eng --oem 3 --psm 11'
        d = pytesseract.image_to_data(img, config=custom_config,output_type = Output.DICT)
        d = seperate(d['text'])
        if len(d)>1:
            return d[1]
        else:
            return d

def date_extract(img,lang_input):
    if lang_input=='english':
        custom_config = r'-l eng --psm 4 --oem 3'
        keywords = ['due','end']
        canny = get_grayscale(img)
        d = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
        #print("date",d['text'])
        d = seperate(d['text'])
        try:
            for i in ['due','end']:
                if i in " ".join(d).lower():
                    return {"due_date":d[-1].split(" ")[-1]}
            else:
                    return {"invoice_date":d[-1].split(" ")[-1]}
        except:
            pass
    elif lang_input=='arabic':
        custom_config = r'-l ara --psm 4 --oem 3'
        keywords = ['due','end']
        canny = get_grayscale(img)
        d = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
        #print("date",d['text'])
        d = seperate(d['text'])
        try:
            for i in ['due','end']:
                if i in " ".join(d).lower():
                    predicted_response['due_date']=d[-1].split(" ")[-1]
                
            else:
                    predicted_response['invoice_date']=d[-1].split(" ")[-1]
        except:
            pass
    

def currency_extract(img,lang_input):
    canny = get_grayscale(img)
    h, w = canny.shape
    if lang_input=='english':
        custom_config = r'-l eng --oem 3 --psm 6 -c tessedit_char_whitelist="$£€¥₹"'
        d = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
        #print("currency",d['text'])
        # d = correction(d['text'])
        d = "".join(d['text'])
        d = "".join(d)
        return d
    elif lang_input=='arabic':
        custom_config = r'-l eng --oem 3 --psm 11 -c tessedit_char_whitelist="د.إ"'
        d = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
        #print("currency",d['text'])
        # d = correction(d['text'])
        d = "".join(d['text'])
        d = "".join(d)
        return d

def col_extract(img):
    img = cv2.resize(img, (int(img.shape[1] + (img.shape[1] * .1)),
                           int(img.shape[0] + (img.shape[0] * .25))),
                     interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    custom_config = r'-l eng --oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-:.$%.,/@& *"'
    d = pytesseract.image_to_data(img_rgb, config=custom_config, output_type=Output.DICT)
    df = pd.DataFrame(d)
    #print(df)
    df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
    try:
        for block in sorted_blocks:
            curr = df1[df1['block_num'] == block]
            sel = curr[curr.text.str.len() > 3]
            # sel = curr
            char_w = (sel.width / sel.text.str.len()).mean()
            prev_par, prev_line, prev_left = 0, 1, 0
            text = ''
            l = []
            for ix, ln in curr.iterrows():
                if prev_par != ln['par_num']:
                    text += '\n'
                    prev_par = ln['par_num']
                    prev_line = ln['line_num']
                    prev_left = 0
                elif prev_line != ln['line_num']:
                    text += '\n'
                    prev_line = ln['line_num']
                    prev_left = 0
                added = 0  # num of spaces that should be added
                if ln['left'] / char_w > prev_left + 1:
                    added = int((ln['left']) / char_w) - prev_left
                    text += ' ' * added
                text += ln['text'] + ' '

                prev_left += len(ln['text']) + added + 1

            text += '\n'
            text = text.split('\n')
            res = [x.strip(' ') for x in text if x.strip(' ')]
        return res[1::]
    except:
        pass


def arabic_col_extract(img):
    custom_config = r"--psm 11 --oem 3"
    return [i for i in pytesseract.image_to_string(img, lang='ara',config=custom_config).split("\n")[1:] if i]

def company_name_extract(img):
    custom_config = r'-l eng --psm 11 --oem 3'
    canny = get_grayscale(img)
    d = pytesseract.image_to_string(canny, config=custom_config)
    #print("company",d)
    d = "".join(d)
    return d

def phone_number(img,lang_input):
    custom_config = r'-l eng --psm 11 --oem 3 -c tessedit_char_whitelist="0123456789+"'
    canny = get_grayscale(img)
    d = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
    #print("phone number",d['text'])
    # d = correction(d['text'])
    d = "".join(d['text'])
    return d
         


def tax_extract(img,lang_input):
    if lang_input == 'english':
        custom_config = r'-l eng --psm 11 --oem 3 -c tessedit_char_whitelist="0123456789%,."'
        canny = get_grayscale(img)
        d = pytesseract.image_to_data(canny, config=custom_config, output_type=Output.DICT)
        #print("hello",d['text'])
        d = seperate(d['text'])
        if len(d)>0:
            d = "".join(d[-1])
            return d
        else:
            return ""
    elif lang_input == 'arabic':
        custom_config = r"--psm 11 --oem 3 -c tessedit_char_whitelist=1234567890%.)("
        img = get_grayscale(img)
        result = pytesseract.image_to_string(img, lang='ara',config=custom_config)
        return [i for i in result.split("\n") if i] if len([i for i in result.split("\n") if i])>1 else result.strip("\n .")
        

def discount_extract(img,lang_input):
    if lang_input=='english':
        custom_config = r'-l eng --psm 11 --oem 3 -c tessedit_char_whitelist="0123456789,."'
        canny = get_grayscale(img)
        d = pytesseract.image_to_data(canny, config=custom_config, output_type=Output.DICT)
        #print("hello",d['text'])
        #d = seperate(d['text'])
        #print(d)
        return "".join(d['text'])
    elif lang_input == 'arabic':
        custom_config = r"--psm 11 --oem 3 -c tessedit_char_whitelist=1234567890%.)("
        img = get_grayscale(img)
        result = pytesseract.image_to_string(img, lang='ara',config=custom_config)
        return [i for i in result.split("\n") if i] if len([i for i in result.split("\n") if i])>1 else result.strip("\n .")


def base64_create(img):
    _, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    jpg_buffer = str(im_b64)[2:len(str(im_b64)) - 1]
    return jpg_buffer




def table(img):
    img = cv2.resize(img, (int(img.shape[1] + (img.shape[1] * .1)),
                           int(img.shape[0] + (img.shape[0] * .25))),
                     interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    custom_config = r'-l eng --oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-:.$%.,/@& *"'
    d = pytesseract.image_to_data(img_rgb, config=custom_config, output_type=Output.DICT)
    df = pd.DataFrame(d)
    #print(df)
    df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    values = []
    headers = []
    sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
    for block in sorted_blocks:
        curr = df1[df1['block_num'] == block]
        sel = curr[curr.text.str.len() > 3]
        # sel = curr
        char_w = (sel.width / sel.text.str.len()).mean()
        prev_par, prev_line, prev_left = 0, 0, 0
        text = ''
        l = []
        for ix, ln in curr.iterrows():
            if prev_par != ln['par_num']:
                text += '\n'
                prev_par = ln['par_num']
                prev_line = ln['line_num']
                prev_left = 0
            elif prev_line != ln['line_num']:
                text += '\n'
                prev_line = ln['line_num']
                prev_left = 0
            added = 0  # num of spaces that should be added
            if ln['left'] / char_w > prev_left + 1:
                added = int((ln['left']) / char_w) - prev_left
                text += ' ' * added
            text += ln['text'] + ' '

            prev_left += len(ln['text']) + added + 1

        text += '\n'
        text = text.split('\n')
        res = [x for x in text if x.strip()]
        z = res[0].split('   ')
        
        headers = [x.strip(' ') for x in z if x.strip()]
        
        for i in range(1, len(res)):
            values.append([x.strip(' ') for x in res[i].split('    ') if x.strip()])
    return (headers, values)

