import pickle


pickled_model = pickle.load(open('textmodel2.pkl', 'rb'))

classes = ['Appliances', 'Automotive', 'Electronics', 'Health Care', 'beauty',
           'food', 'industry']

def categorize(final_text):
    conf = pickled_model.predict_proba([final_text])
    confidence_classes = {classes[i]: conf[0][i] for i in range(len(conf[0]))}
    #print(confidence_classes)
    max_conf = max(conf[0])
    max_class = sorted(confidence_classes, key=lambda x: confidence_classes[x])[-1]
    if max(conf[0]) < 0.5:
        sorted(confidence_classes, key=lambda x: confidence_classes[x])[-1]
        return "miscellaneous",100
    else:
        return str(pickled_model.predict([final_text])[0]),max(conf[0]) * 100