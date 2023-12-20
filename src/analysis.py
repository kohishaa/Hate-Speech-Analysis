import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from lime.lime_text import LimeTextExplainer
import joblib
import random
from streamlit_option_menu import option_menu
from gensim.models import Word2Vec
#st.header('*Hate Speech Classification*')

#Countvector
#model_count_lr= joblib.load('finalCapstone/model/count_lr.pkl')
model_count_dt= joblib.load('finalCapstone/model/count_dt.pkl')
model_count_svm= joblib.load('finalCapstone/model/count_svm.pkl')
model_count_nb= joblib.load('finalCapstone/model/count_nb.pkl')
model_count_rf= joblib.load('finalCapstone/model/count_rf.pkl')

#Word2vec

word2vec = joblib.load('finalCapstone/model/word2vec.pkl')
model_wrdevec_dt= joblib.load('finalCapstone/model/word2vecdt.pkl')
model_wrdevec_svm= joblib.load('finalCapstone/model/word2vec_svm.pkl')
model_wrdevec_gb= joblib.load('finalCapstone/model/word2vec_gb.pkl')
model_wrdevec_nb= joblib.load('finalCapstone/model/word2veclr.pkl')

#Tfidf
model_gb= joblib.load('finalCapstone/model/gradboostbalanced.pkl')
model_dt= joblib.load('finalCapstone/model/model_dtbalanced.pkl')
tfidf_vect=joblib.load('finalCapstone/model/tfidf50000.pkl')
model_nb= joblib.load('finalCapstone/model/naive_bayesbalanced.pkl')
model_svm= joblib.load('finalCapstone/model/svcmodelbalanced.pkl')

#GloVe



data=pd.read_csv('finalCapstone/src/data.csv')

data_clean=data.dropna(subset=['clean_comment_string1'])
#data_clean=data_clean[:36000]
data_clean['target']=data_clean['target'].replace({'NoHate':0,'Hate':1})
x=data_clean['clean_comment_string1']
y=data_clean[['target']]
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=12)

def show():
    selected = st.sidebar.selectbox("Select a Function", ("Decision Trees","Naive Bayes","SVM","Gradient Boosting"))
    return selected

def vectorize(sentence):
    words = sentence.split()
    words_vecs = [word2vec.wv[word] for word in words if word in word2vec.wv]
    if len(words_vecs) == 0:
        return np.zeros(100)
    words_vecs = np.array(words_vecs)
    return words_vecs.mean(axis=0)


def renderPage():
    model = show()
    print(model)
    st.title("Hate Speech Classification😊")
    #components.html("""<hr style="height:3px;border:none;color:#333;background-color:#333; margin-bottom: 2px" /> """)
    st.subheader("User Input Text Analysis")
    st.text("Analyzing text data given by the user ")
    st.text("Provide the Text with Less than 30 Words ")
    userText = st.text_input('User Input')
    st.text("")
    type = st.selectbox(
     'Type of analysis',
     ('Word2Vec', 'TFIDF','GloVe','Count'))
    st.text("")
    if st.button('Predict'):
        if(userText!="" and type!=None):
            if type=='TFIDF':
                if model=='Decision Trees':
                    model_prediction_tfidf(userText,tfidf_vect,model_dt)
                elif model=='SVM':
                    model_prediction_tfidf(userText,tfidf_vect,model_svm)
                elif model=='Gradient Boosting':
                    model_prediction_tfidf(userText,tfidf_vect,model_gb)
                elif model=='Naive Bayes':
                    model_prediction_tfidf(userText,tfidf_vect,model_nb)

            elif type=='Word2Vec':
                if model=='Decision Trees':
                    model_prediction_wordvec(userText,model_wrdevec_dt)
                elif model=='SVM':
                    model_prediction_wordvec(userText,model_wrdevec_svm)
                elif model=='Gradient Boosting':
                    model_prediction_wordvec(userText,model_wrdevec_gb)
                elif model=='Naive Bayes':
                    model_prediction_wordvec(userText,model_wrdevec_nb)
            elif type=='Count':
                if model=='Decision Trees':
                    model_prediction_count(userText,model_count_dt)
                elif model=='SVM':
                    model_prediction_count(userText,model_count_svm)
                elif model=='Gradient Boosting':
                    model_prediction_count(userText,model_count_rf)
                elif model=='Naive Bayes':
                    model_prediction_count(userText,model_count_nb)


def model_prediction_tfidf(text,vect,algo):
    binary_model = Pipeline([('vectorizer', vect), ('classifier', algo)])
    result = binary_model.predict([text])
    print(result)
    print_result(result,binary_model,text)

def model_prediction_wordvec(text,algo):
    arraytext=np.array(vectorize(text))
    reshaped_data=arraytext.reshape(1,-1)
    result = algo.predict(reshaped_data)
    print("Result")
    print(result)
    print_result_wordvec(result,algo,text)

def model_prediction_count(text,algo):
    result = algo.predict([text])
    
    print(result)
    print_result(result,algo,text)

def limeword(text):
    print(type(text))
    print(text)
    arraytext=np.array(vectorize(text))
    reshaped_data=arraytext.reshape(1,-1)
    # Interpretation of Result
    st.write("""#### Result Interpretation:""")
    probabi=model_wrdevec_dt.predict_proba(reshaped_data)
    #format_pred = np.concatenate([1.0-pred, pred], axis=1)

    return probabi



def print_result_wordvec(result,binary_model,text):
    if result.astype(int) == 0:
            result_text = "Hate Speech"
    else:
        result_text = "Non-Hate Speech"
        
    st.write(" ##### Result: ", result_text)
    # print("BInary Text")
    # print(text)
    # arraytext=np.array(vectorize(text))
    # reshaped_data=arraytext.reshape(1,-1)
    # # Interpretation of Result
    # st.write("""#### Result Interpretation:""")
    # probabi=binary_model.predict_proba(reshaped_data)
    # print(probabi)
    # binary_explainer = LimeTextExplainer(class_names={"Hate":0, "Non-Hate":1})
    # max_features = X_train.str.split().map(lambda x: len(x)).max()
    
    
    # random.seed(13)
    # idx = random.randint(0, len(X_test))
    
    # bin_exp = binary_explainer.explain_instance(
    #     str(text), limeword, num_features=max_features
    #     )
    # print(bin_exp.as_list())
    # components.html(bin_exp.as_html(), height=800)


def print_result(result,binary_model,text):
    if result.astype(int) == 0:
            result_text = "Hate Speech"
    else:
        result_text = "Non-Hate Speech"
        
    st.write(" ##### Result: ", result_text)
    print("BInary Text")
    print(text)
    
    # Interpretation of Result
    st.write("""#### Result Interpretation:""")
    probabi=binary_model.predict_proba([text])
    print(probabi)
    binary_explainer = LimeTextExplainer(class_names={"Hate":0, "Non-Hate":1})
    max_features = X_train.str.split().map(lambda x: len(x)).max()
    
    
    random.seed(13)
    idx = random.randint(0, len(X_test))
    
    bin_exp = binary_explainer.explain_instance(
        text, binary_model.predict_proba, num_features=max_features
        )
    print(bin_exp)
    components.html(bin_exp.as_html(), height=800)


      
renderPage()


            