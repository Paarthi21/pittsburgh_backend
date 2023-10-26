from textblob import TextBlob



def correction(text):
    text = "".join(text)
    text = TextBlob(text).correct()
    return str(text)



def seperate(arr):
    array=[]
    temp=""
    seq=False
    for i in arr:
        pre_seq = seq
        if i:
            seq = True
            temp+=i+" "
        else:
            seq = False
            temp = ""
        if pre_seq and seq:
            array[-1]+=temp
            temp = ''
        if pre_seq == False and seq == True:  
            try:
                array[-1] = array[-1].strip(" ")
            except:
                pass
            array.append(temp)
            temp=""
    try:
        array[-1] = array[-1].strip(" ")
    except:
        pass
    return array

    # return " ".join(array[1:]) if len(array)>1 else array[0].split(" ")[-1]