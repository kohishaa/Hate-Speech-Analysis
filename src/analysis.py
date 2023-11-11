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

data=pd.read_csv('./src/data.csv')

data_clean=data.dropna(subset=['clean_comment_string1'])
data_clean=data_clean[:1000]
data_clean['target']=data_clean['target'].replace({'NoHate':0,'Hate':1})
x=data_clean['clean_comment_string1']
y=data_clean[['target']]
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=12)
tfidf_vect = TfidfVectorizer()
#tfidf_vect = TfidfVectorizer(max_features=5000)
tfidf_vect.fit(data_clean['clean_comment_string1'].astype(str))
x_train_tfidf = tfidf_vect.transform(X_train)
x_test_tfidf = tfidf_vect.transform(X_test)

import random
st.header('**Hate Speech Classification**')
st.write('---')

menu = st.sidebar.selectbox("Select a Function", ("Decision Trees","Naive Bayes","SVM","Gradient Boosting"))
st.header('*Model Evaluation*')
st.write("""##### Try it out yourself!""")
#binary_text = st.text_area("Classify Using The Binary Model:", "Enter Text")  
svm_binary = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
svm_binary.fit(x_train_tfidf, Y_train)

# predict the labels on validation dataset
y1_pred = svm_binary.predict(x_test_tfidf)
# Classifier Evaluation
#classifier_evaluation(y1_pred, Y_test)
#User Input
#st.write("""##### Try it out yourself!""")
binary_text = st.text_area("Classify Using The Binary Model:", "Enter Text")  
#Clean the Text
if menu=="Decision Trees":
    model= joblib.load('model/gradboost.pkl')


    if st.checkbox('Apply Classification Model'):
        # Preparation for Classifying User Input
        binary_model = Pipeline([('vectorizer', tfidf_vect), ('classifier', model)])
        
        
        # Generate Result
        result = binary_model.predict([binary_text])
        
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
        
        components.html(bin_exp.as_html(), height=800)


if menu=="SVM":
    model= joblib.load('model/svcmodel.pkl')


    if st.checkbox('Apply Classification Model'):
        # Preparation for Classifying User Input
        binary_model = Pipeline([('vectorizer', tfidf_vect), ('classifier', model)])
        
        
        # Generate Result
        result = binary_model.predict([binary_text])
        
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
        
        components.html(bin_exp.as_html(), height=800)

if menu=="Gradient Boosting":
    model= joblib.load('model/gradboost.pkl')


    if st.checkbox('Apply Classification Model'):
        # Preparation for Classifying User Input
        binary_model = Pipeline([('vectorizer', tfidf_vect), ('classifier', model)])
        
        
        # Generate Result
        result = binary_model.predict([binary_text])
        
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
        
        components.html(bin_exp.as_html(), height=800)
if menu=="Naive Bayes":
    model= joblib.load('model/naive_bayes.pkl')


    if st.checkbox('Apply Classification Model'):
        # Preparation for Classifying User Input
        binary_model = Pipeline([('vectorizer', tfidf_vect), ('classifier', model)])
        
        
        # Generate Result
        result = binary_model.predict([binary_text])
        
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
        
        components.html(bin_exp.as_html(), height=800)