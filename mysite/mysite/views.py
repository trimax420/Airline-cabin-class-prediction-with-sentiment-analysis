from django.http import HttpResponse
from django.shortcuts import render
import pathlib
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
from os import path
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from wordcloud import ImageColorGenerator
import re
import pickle
import joblib
import spacy
import nltk
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
import string

from nltk import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime
import warnings
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from bs4 import BeautifulSoup
from gensim.models import CoherenceModel
from gensim import corpora, models
import plotly.express as px
import base64
from mysite.forms import twitterForm , airlineForm
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
from io import StringIO
import io


oe = OrdinalEncoder()
object_cols = ["'TICKETING_AIRLINE'","'COUNTRY'","'TRANSACTION_TYPE'","'TRIP_TYPE'","'MARKETING_AIRLINE'","'MARKETING_AIRLINE_CD'","'CABIN'","'ORIGIN'","'DESTINATION'"]
option = pd.read_csv(DATA_PATH.joinpath("travelverse-dataset.csv"))
sample = pd.read_csv(DATA_PATH.joinpath("travelverse-dataset.csv"))
sample[object_cols] = oe.fit_transform(sample[object_cols])
X = sample[["'TICKETING_AIRLINE'","'COUNTRY'","'ORIGIN'","'DESTINATION'"]]
y = sample["'CABIN'"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(128,64,32),verbose=True,max_iter=1)
mlp.fit(X_train,y_train)
tf_1=sample["'ORIGIN'"].unique()
df_1=option["'ORIGIN'"].unique()
tf_2=sample["'COUNTRY'"].unique()
df_2=option["'COUNTRY'"].unique()
tf_3=sample["'TICKETING_AIRLINE'"].unique()
df_3=option["'TICKETING_AIRLINE'"].unique()
tf_4=sample["'DESTINATION'"].unique()
df_4=option["'DESTINATION'"].unique()
tf_5=sample["'CABIN'"].unique()
df_5=option["'CABIN'"].unique()
origin_df = list(zip(df_1,tf_1))
origin_df = pd.DataFrame(origin_df,columns=['name','int'])
country_df = list(zip(df_2,tf_2))
country_df = pd.DataFrame(country_df,columns=['name','int'])
ticketing_airline_df = list(zip(df_3,tf_3))
ticketing_airline_df = pd.DataFrame(ticketing_airline_df,columns=['name','int'])
destination_df = list(zip(df_4,tf_4))
destination_df = pd.DataFrame(destination_df,columns=['name','int'])
cabin_df = list(zip(df_5,tf_5))
cabin_df = pd.DataFrame(cabin_df,columns=['name','int'])



def twitter(request):
    def predict_topic(tweet):
        lda_model4 = models.LdaModel.load('datasets/lda_model4.model')

        
        def preprocess(text):
            stopwords = set(STOPWORDS)
            stopwords.update(["american", "air", "airline", "thank", "united", "us", "airways", "virgin", "america", "jetblue", "youre", "extremely",
                              "usairway", "usairways", "flight", "americanair", "southwestair", "southwestairlines", "arbitrarily", "dream", "crazy",
                              "southwestairway", "southwestairways", "virginamerica", "really", "will", "going", "thanks", "thankyou", "passengersdont",
                              "please", "got", "let", "take", "help", "already", "never", "now", "told", "guy", "new", "sure", "still", "amp", "continue",
                              "plane", "tell", "ye", "trying", "yes", "guy", "much", "appreciate", "thx", "back", "ok", "good", "credit", "aacom",
                              "flying", "love", "great", "awesome", "see", "nice", "alway", "httptcojwl26g6lrw", "dontflythem", "motherinlaw", "night",
                              "nogearnotraining", "seriously", "didnt", "coudnt", "cant", "wont", "dont", "wat", "buffaloniagara", "hasshe", "morning",
                              "woulda", "people", "try", "youve", "youd", "yours", "flightled", "tomorrow", "today", "wat", "jfkyou", "flite", "cause",
                              "flightr", "flight", "need", "hours", "nooooo", "like", "doesnt", "right", "talk", "tweet", "mention", "pbijfk", "ridiculuous",
                              "wasnt", "suppose", "want", "understand", "come", "work", "worse", "treat", "think", "know", "worst", "paulo", "staduim",
                              "wouldnt", "stay", "away", "wont", "werent", "happen", "sorry", "havent", "tonight", "drive", "life", "thing", "aa951",
                              "whats", "theyre", "better", "thats", "allow", "hope", "stop", "cool", "niece", "happy", "word", "customercant",
                              "suck", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "weekend", "ruin", "shouldnt",
                              "miami", "los angeles", "new york", "chicago", "dallas", "apparently", "itover", "someones", "savannah", "lucymay",
                              "betterother", "instead", "look", "hopefully", "yesterday", "antonio", "unacceptable", "folks", "record", 'arent',
                              "miss", "hang", "wrong", "stick", "grind", "tarmac", "theres", "forget", "terrible", "clothe", "terrible", "break",
                              "actually", "frustrate", "correct", "ridiculous", "expect", "different", "pathetic", "bother", "follow", "fault",
                              "impossible", "point", "cover", "person", "ask", "speak", "things", "earlier", "mean", "select", "minutes",
                              "unite", "horrible", "country", "leave", "speak", "apologize", "faster", "hop", "confuse", "lose", "flightd", "hear",
                              "literally", "years", "surprise", "bump", "fail", "compensate", "hand", "helpful", "upset", "friend", "excuse", "claim",
                              "situation", "multiple", "weather", "choose", "company", "believe", "question", "kick", "anymore", "awful", "delta",
                              "dozen", "medical", "completely", "finally", "waste", "shock", "annoy", "maybe", "strand", "mess", "finally",
                              "plan", "place", "apology", "center", "plan", "twitter", "promise", "prefer", "count", "maybe", "shock", "longer", "meet",
                              "important", "drop"])
            
            r = re.compile(r'(?<=\@)(\w+)')
            ra = re.compile(r'(?<=\#)(\w+)')  
            ro = re.compile(r'(flt\d*)')  
            names = r.findall(text.lower())
            hashtag = ra.findall(text.lower())
            flight = ro.findall(text.lower())
            lmtzr = WordNetLemmatizer()

            def stem_tokens(tokens, lemmatize):
                lemmatized = []
                for item in tokens:
                    lemmatized.append(lmtzr.lemmatize(item, 'v'))
                return lemmatized

            def deEmojify(inputString):
                return inputString.encode('ascii', 'ignore').decode('ascii')

           
            text = deEmojify(text)
            soup = BeautifulSoup(text)
            text = soup.get_text()
            text = "".join([ch.lower()
                           for ch in text if ch not in string.punctuation])
            tokens = nltk.word_tokenize(text)
           
            tokens = [ch for ch in tokens if len(ch) > 4]
           
            tokens = [ch for ch in tokens if len(ch) <= 15]
            lemm = stem_tokens(tokens, lmtzr)
            lemstop = [i for i in lemm if i not in stopwords]
            lemstopcl = [i for i in lemstop if i not in names]
            lemstopcl = [i for i in lemstopcl if i not in hashtag]
            lemstopcl = [i for i in lemstopcl if i not in flight]
            lemstopcl = [i for i in lemstopcl if not i.isdigit()]
         
            return lemstopcl

        id2word = corpora.Dictionary.load("datasets/id2word.dict")

        bow_vector = id2word.doc2bow(preprocess(tweet))
        result = lda_model4.get_document_topics(bow_vector)
        resultdict = dict(result)
        orddict = sorted(resultdict, key=resultdict.get, reverse=True)

        Keymax = 1
        if resultdict[orddict[0]]-resultdict[orddict[1]] <= .08:
            Keymax += orddict[1]
            Predicted_topic = orddict[1]+1
            Probability_Score = resultdict[orddict[0]]
        else:
            Keymax += orddict[0]
            Predicted_topic = orddict[0]+1
            Probability_Score = resultdict[orddict[0]]

        if Keymax == 1:
            response = 'Delay and Customer Service'
        elif Keymax == 2:
            response = 'Baggage Issue'
        elif Keymax == 3:
            response = 'Reschedule and Refund'
        elif Keymax == 4:
            response = 'Phone and Online Booking'
        elif Keymax == 5:
            response = 'Reservation Issue'
        elif Keymax == 6:
            response = 'Seating Preferences'
        elif Keymax == 7:
            response = 'Extra Charges'
        else:
            response = 'Customer Experience'
        print("\n")

        df = pd.DataFrame(resultdict.items())
        df[0] = df[0]+1

        if Keymax == orddict[1]+1:
            temp = df.iloc[orddict[0], 1]
            df.iloc[orddict[0], [1]] = df.iloc[Keymax-1, 1]
            df.iloc[Keymax-1, [1]] = temp
        else:
            df[1] = df[1]
        
        x=df[0]
        y=df[1]
        colors = ['#3f5994', '#8cbcfb', '#c9e5fa', '#f4dad0',
                  "#f8bc99", '#f3aa84', '#f49c6c', "#c48771"]
       

        return response,Predicted_topic,Probability_Score,Keymax,x,y,colors
            
    
    if request.method == 'POST':
        twitter=request.POST['title']
        response,Predicted_topic,Probability_Score,Keymax,x,y,colors=predict_topic(twitter)
        plt.style.use('ggplot')
        plt.figure()
        plt.title("Topic Probability Score", size=12)
        plt.xticks(rotation=20, size=15)
        plt.yticks(size=10)
        ax = sns.barplot(x, y, palette=colors)
        plt.ylabel('Probability Score', fontsize=12)
        plt.xlabel('Topics', fontsize=20)
        plt.figsize=(25, 25)
        imgdata = StringIO()
        plt.savefig(imgdata, format='svg')
        imgdata.seek(0)
        data = imgdata.getvalue()

        return render(request,"twitter_output.html",{'twitter_html':data,'response':response,'Predicted_topic':Predicted_topic,'probability_score':Probability_Score,'tweet':twitter})
    else:
        twitter = twitterForm()  
        return render(request,"twitter.html",{'form':twitter})    
    
    


def airline(request):
    form = airlineForm() 
    if request.method =="POST":
        form = airlineForm(request.POST) 
        origin=request.POST.get('origin')
        country=request.POST.get('country')
        ticketing_airline=request.POST.get('ticketing_airline')
        destination=request.POST.get('destination')
        
         
        origin_int=origin_df.int.loc[origin_df.name == origin]
        destination_int=destination_df.int.loc[destination_df.name == destination]
        ticketing_airline_int=ticketing_airline_df.int.loc[ticketing_airline_df.name == ticketing_airline]
        country_int=country_df.int.loc[country_df.name == country]
        cabin_int=mlp.predict([[ticketing_airline_int.item(),country_int.item(),origin_int.item(),destination_int.item()]])
        print(cabin_int)
        if cabin_int == 0 :
            cabin = "ECONOMY CLASS"
        if cabin_int == 1 :
            cabin = "PREMIUM CLASS"
        return render(request,'airline_output.html',{'cabin':cabin})

    else:
        form = airlineForm()
        origin_choice = list((df_1))
        country_choice = list((df_2))
        ticketing_airline_choice = list((df_3))
        destination_choice = list((df_4))
        return render(request,'airline.html',{'form':form,'origin_choice':origin_choice,'country_choice':country_choice,'ticketing_airline_choice':ticketing_airline_choice,'destination_choice':destination_choice})


def index(request):

    return render(request,'index.html')

def about(request):

    return render(request,'about.html')    
