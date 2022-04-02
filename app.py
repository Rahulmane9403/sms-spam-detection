from unittest import result
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import pandas as pd 
from transformers import AutoTokenizer, AutoModel , AutoConfig 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 



ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    #text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

#tfidf = pickle.load(open('vectorizer.pkl','rb'))
#model = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Dectection")

input_sms = st.text_area("Enter the message")
if st.button('Predict'):

    # 1. preprocess
   # transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    # no_characters = len(input_sms)
    # no_words =1# len(nltk.word_tokenize(input_sms))
    # no_sentences =1# len(nltk.sent_tokenize(input_sms))
    # testing_text_df = pd.DataFrame(vector_input.toarray())
    # testing_text_df['no_words'] = no_words
    # testing_text_df['no_characters'] =  no_characters
    # testing_text_df['no_sentences'] = no_sentences
    
    # result = model.predict(testing_text_df.to_numpy())
    tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer/")
    model = AutoModelForSequenceClassification.from_pretrained("./my_model/")
    dict_samp = tokenizer(input_sms)
    outputs = model(torch.unsqueeze(torch.tensor(dict_samp['input_ids']), dim = 0 ), torch.unsqueeze(torch.tensor(dict_samp['attention_mask']),0))
    result = 1 if outputs.logits.argmax()==1   else 0


    # 4. Display
    if result == 1:
        st.header("Spam")   
    else:
        st.header(" ham")
