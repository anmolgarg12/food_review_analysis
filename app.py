# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 22:44:03 2021

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 17:19:41 2021

@author: ASUS
"""
import numpy as np
import pandas as pd
import nltk                                                           #nltk-natural language tool kit
import re                                                             #re-regular expression
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from flask import Flask,render_template,url_for,request

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    data=pd.read_csv('C:/Users/ASUS/Desktop/nlp/Restaurant_Reviews.tsv', delimiter = '\t')
    corpus = []
    for i in range(len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])                                                       #to remove alpha numeric words
    review = review.lower()                                                                                    #to convert data into lowercase
    review = review.split()                                                                                    #tokenization-Tokenization in NLP is the process by which a large quantity of text is divided into smaller parts called tokens
    all_stopwords = stopwords.words('english')                                                                 #Stopwords are the English words which does not add much meaning to a sentence, like the,have etc.
    all_stopwords.remove('not')  #remove negative word 'not' as it is closest word to help determine whether the review is good or not 
    review = [stemmer.stem(word) for word in review if not word in set(all_stopwords)]                         #Stemming is the process of reducing inflection in words to their root forms such as mapping a group of words to the same stem.
    review = ' '.join(review)
    corpus.append(review)
    
    cv = CountVectorizer()
    X = cv.fit_transform(corpus).toarray()
    y = data.iloc[:, -1].values
    
    pickle.dump(cv, open('tranform.pkl', 'wb'))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
     
    mnb = MultinomialNB(alpha=2)                               #random_state- to set seed value
    mnb.fit(X_train,y_train)
    y_pred_mnb=mnb.predict(X_test) 
    filename = 'nlp_model.pkl'
    pickle.dump(mnb, open(filename, 'wb')) 
    '''
    if request.method == 'POST':
        message = request.form['message']
        message = re.sub("[^a-zA-Z]"," ",message)
        message = message.lower().split()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not') 
        message = [stemmer.stem(word) for word in message if word not in set(all_stopwords)]
        message = " ".join(message)
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)