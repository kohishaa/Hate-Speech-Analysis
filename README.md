###Hate Speech Analysis Project

This project builds a system to detect and classify hate speech in text using a combination of machine learning and deep learning models. It leverages multiple datasets and provides interpretable results with the LIME algorithm.

##Key Components

#Data Collection: Diverse datasets sourced from platforms like news comments, Wikipedia, and social media, ensuring a balanced representation of hate speech and non-hate speech.
#Text Preprocessing: Includes tokenization, stemming, lemmatization, and removal of unwanted elements like URLs and stopwords.
#Modeling: Implements models like Support Vector Machines (SVM), Naive Bayes, Gradient Boosting, CNN-LSTM, and BERT for hate speech classification.
#Interpretability: Uses LIME to explain and visualize model predictions, making the results understandable.

##Explainable AI
Used the LIME (Local Interpretable Model-agnostic Explanations) algorithm to make model predictions interpretable by showing important features that contributed to the decision.

##Model Evaluation
The models are evaluated using:

Precision
Recall
F1 Score
Accuracy
Confusion Matrix
Deep learning models were trained over multiple epochs, with BERT and CNN-LSTM architectures optimized for text classification.

##Results
The results show that the models effectively classify hate speech and non-hate speech, providing interpretable predictions.

##Streamlit App
The predictions of each model are visualized and analyzed using a Streamlit app.

##Getting Started

Clone the repository.
Install dependencies.
Run the project through the provided notebook.
