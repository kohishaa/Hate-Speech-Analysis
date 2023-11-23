import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
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

model_gb= joblib.load('model/gradboost.pkl')
model_dt= joblib.load('model/model_dtbalanced.pkl')
tfidf_vect=joblib.load('model/tfidf50000.pkl')
model_nb= joblib.load('model/naive_bayes.pkl')
model_svm= joblib.load('model/svcmodel.pkl')
count_vect= joblib.load('model/count.pkl')

word2vec = Word2Vec.load('model/word2vec_model.bin')
model_wrdevec_dt= joblib.load('model/model_dtword2vec.pkl')
model_wrdevec_svm= joblib.load('model/model_svmword2vec.pkl')

data=pd.read_csv('./src/data.csv')

data_clean=data.dropna(subset=['clean_comment_string1'])
#data_clean=data_clean[:36000]
data_clean['target']=data_clean['target'].replace({'NoHate':0,'Hate':1})
x=data_clean['clean_comment_string1']
y=data_clean[['target']]
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=12)

def show():
    selected = st.sidebar.selectbox("Select a Function", ("Decision Trees","Naive Bayes","SVM","Gradient Boosting"))
    return selected



def renderPage():
    model = show()
    print(model)
    st.title("Hate Speech Classification😊")
    #components.html("""<hr style="height:3px;border:none;color:#333;background-color:#333; margin-bottom: 2px" /> """)
    st.subheader("User Input Text Analysis")
    st.text("Analyzing text data given by the user ")
    st.text("Provide the Text with Less than 30 Words ")
    userText = st.text_input('User Input', placeholder='Input text HERE')
    st.text("")
    type = st.selectbox(
     'Type of analysis',
     ('Word2Vec', 'TFIDF','GloVe'))
    st.text("")
    if st.button('Predict'):
        if(userText!="" and type!=None):
            if type=='TFIDF':
                if model=='Decision Trees':
                    model_prediction(userText,tfidf_vect,model_dt)
                elif model=='SVM':
                    model_prediction(userText,tfidf_vect,model_svm)
                elif model=='Gradient Boosting':
                    model_prediction(userText,tfidf_vect,model_gb)
                elif model=='Naive_Bayes':
                    model_prediction(userText,tfidf_vect,model_nb)

            elif type=='Word2Vec':
                if model=='Decision Trees':
                    model_prediction(userText,tfidf_vect,model_dt)
                elif model=='SVM':
                    model_prediction(userText,tfidf_vect,model_svm)
                elif model=='Gradient Boosting':
                    model_prediction(userText,tfidf_vect,model_gb)
                elif model=='Naive_Bayes':
                    model_prediction(userText,tfidf_vect,model_nb)

                    


def model_prediction(text,vect,algo):
    binary_model = Pipeline([('vectorizer', vect), ('classifier', algo)])
    result = binary_model.predict([text])
    print(result)
    print_result(result,binary_model,text)

def print_result(result,binary_model,binary_text):
    if result.astype(int) == 0:
            result_text = "Hate Speech"
    else:
        result_text = "Non-Hate Speech"
        
    st.write(" ##### Result: ", result_text)
    
    # Interpretation of Result
    st.write("""#### Result Interpretation:""")
    binary_model.predict_proba([binary_text])
    binary_explainer = LimeTextExplainer(class_names={"Hate":0, "Non-Hate":1})
    max_features = X_train.str.split().map(lambda x: len(x)).max()
    
    
    random.seed(13)
    idx = random.randint(0, len(X_test))
    
    bin_exp = binary_explainer.explain_instance(
        binary_text, binary_model.predict_proba, num_features=max_features
        )
    print(bin_exp)
    components.html(bin_exp.as_html(), height=800)

            
renderPage()
