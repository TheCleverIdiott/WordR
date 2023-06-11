from flask import Flask, render_template
import pandas as pd
import json
import plotly
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import tensorflow
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import nltk
import re

app = Flask(__name__)

@app.route("/work")
def work():
    return render_template('howitworks.html')

###########################Main Page ###############################
@app.route("/")
def main():
    return render_template('index.html')


@app.route("/analysis")
def analysis():
    
    return render_template('analysis.html')


def cleantext(txt,sw,ps):
  lst1="".join([i for i in txt if i not in string.punctuation])
  lst2=re.split('\W+',txt)
  lst3=[i for i in lst2 if i not in sw]
  lst4=[ps.stem(i) for i in lst3]
  return lst4

#1st sec#
@app.route("/social")
def socialMedia():
    
    #sec1p1
    nltk.download('stopwords')

    sw=nltk.corpus.stopwords.words("english")
    ps=nltk.PorterStemmer()
    tokenizer=Tokenizer()
    model_load=tensorflow.keras.models.load_model("static/models/sentiment.h5")

    df1=pd.read_csv("static/dataset/section1/twitter_tweets.csv",encoding= 'unicode_escape')
    df1["Clean"]=df1["Text"].apply(lambda x:cleantext(x,sw,ps))
    tokenizer.fit_on_texts(df1["Clean"])
    x1=tokenizer.texts_to_sequences(df1["Clean"])
    x_padded1=pad_sequences(x1,50)
    sentiments1=model_load.predict(x_padded1)
    finalsentiments1=[]
    for it in sentiments1:
        if(it<0.3):
            finalsentiments1.append(-1)
        elif(it>=0.3 and it<=0.4):
            finalsentiments1.append(0)
        elif(it>0.4):
            finalsentiments1.append(1)
    twitterct1=0
    twitterct2=0
    twitterct3=0
    for i1 in finalsentiments1:
        if(i1==-1):
            twitterct1+=1
        elif(i1==0):
            twitterct2+=1
        elif(i1==1):
            twitterct3+=1
    twittercounts=[twitterct1,twitterct2,twitterct3]
    plotdf1=pd.DataFrame({"Twitter Tweet Sentiments":finalsentiments1})
    plot1=px.histogram(plotdf1,x="Twitter Tweet Sentiments",title="")
    fig1 = plot1
    #pl2
    labels=["Negative","Neutral","Positive"]
    values=twittercounts
    fig2 =go.Figure(data=[go.Pie(labels=labels,values=values)])
  #p13

    df2=pd.read_csv("static/dataset/section1/reddit_posts.csv",encoding='unicode_escape')
    df2["Clean"]=df2["ï»¿Text"].apply(lambda x:cleantext(x,sw,ps))
    tokenizer.fit_on_texts(df2["Clean"])
    x2=tokenizer.texts_to_sequences(df2["Clean"])
    x_padded2=pad_sequences(x2,50)
    sentiments2=model_load.predict(x_padded2)
    finalsentiments2=[]
    for it in sentiments2:
        if(it<0.3):
            finalsentiments2.append(-1)
        elif(it>=0.3 and it<=0.4):
            finalsentiments2.append(0)
        elif(it>0.4):
            finalsentiments2.append(1)
    redditct1=0
    redditct2=0
    redditct3=0
    for i2 in finalsentiments2:
        if(i2==-1):
            redditct1+=1
        elif(i2==0):
            redditct2+=1
        elif(i2==1):
            redditct3+=1
    redditcounts=[redditct1,redditct2,redditct3]
    plotdf2=pd.DataFrame({"Reddit Post Sentiments":finalsentiments2})
    plot3=px.histogram(plotdf2,x="Reddit Post Sentiments",title="")
    fig3 = plot3

    labels=["Negative","Neutral","Positive"]
    values=redditcounts
    plot4=go.Figure(data=[go.Pie(labels=labels,values=values)])
    fig4 = plot4
    
    df3=pd.read_csv("static/dataset/section1/linkedin_content.csv",encoding='unicode_escape')

    df3["Clean"]=df3["Text"].apply(lambda x:cleantext(x,sw,ps))
    tokenizer.fit_on_texts(df3["Clean"])
    x3=tokenizer.texts_to_sequences(df3["Clean"])
    x_padded3=pad_sequences(x3,50)
    sentiments3=model_load.predict(x_padded3)
    finalsentiments3=[]
    for it in sentiments3:
        if(it<0.3):
            finalsentiments3.append(-1)
        elif(it>=0.3 and it<=0.4):
            finalsentiments3.append(0)
        elif(it>0.4):
            finalsentiments3.append(1)
    linkedinct1=0
    linkedinct2=0
    linkedinct3=0
    for i3 in finalsentiments3:
        if(i3==-1):
            linkedinct1+=1
        elif(i3==0):
            linkedinct2+=1
        elif(i3==1):
            linkedinct3+=1
    linkedincounts=[linkedinct1,linkedinct2,linkedinct3]
    plotdf3=pd.DataFrame({"LinkedIn Content Sentiments":finalsentiments3})
    plot5=px.histogram(plotdf3,x="LinkedIn Content Sentiments",title="")
    fig5 = plot5
   #p16
    labels=["Negative","Neutral","Positive"]
    values=linkedincounts
    plot6=go.Figure(data=[go.Pie(labels=labels,values=values)])
    fig6 = plot6

    
    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON5 = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON6 = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    return render_template('socialMedia.html', graphJSON1=graphJSON1,graphJSON2=graphJSON2,graphJSON3=graphJSON3, graphJSON4  = graphJSON4,graphJSON5=graphJSON5,graphJSON6=graphJSON6)

#s13
@app.route("/sale")
def dataSale():
    df1=pd.read_csv(r"static/dataset/section3/bbd_customer_data.csv")
    
    plot1 = px.bar(df1,x="Age",y="Are you satisfied with the products during 'The Big Billion Days'",color="Have you shopped during 'The Big Billion Days'?")
    plot1.update_layout(
    legend=dict(
          orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
   
    ),
    autosize=False,
    width=550,
    height=400,
    yaxis=dict(
        title_text="Are you satisfied with the products during 'The Big Billion Days'",
        titlefont=dict(size=10),
        )
    )


    plot2 = px.histogram(df1,x="Age",y="Are you satisfied with the products during 'The Big Billion Days'",color="Profession", height=400)
    plot2.update_layout(
            legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
    
        ),
        autosize=False,
        width=550,
        height=400,
        yaxis=dict(
            title_text="Are you satisfied with the products during 'The Big Billion Days'",
            titlefont=dict(size=10),
            )
        )
    plot3 = px.histogram(df1,x="Would you consider joining Flipkart Plus Membership to avail additional benefits from 'The Big Billion Days'?",color="Have you shopped during 'The Big Billion Days'?")
    plot3.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
    
        ),
        autosize=False,
        width=550,
        height=400,
        xaxis=dict(
            title_text="Would you consider joining Flipkart Plus Membership to avail additional benefits from 'The Big Billion Days'?",
            titlefont=dict(size=10),
            )
        )
    plot4 = px.bar(df1,x="How satisfied were you while purchasing from Flipkart",color="How often do you shop from Flipkart?")
    plot4.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1

    ),
    autosize=False,
    width=550,
    height=400,
    xaxis=dict(
        title_text="How satisfied were you while purchasing from Flipkart",
        titlefont=dict(size=10),
        )
    )
    df2=pd.read_csv(r"static/dataset/section3/bbd_product_analysis.csv")

    plot5 = px.scatter(df2,x="Product_ID",y="Product_Category",color="City_Category")

    plot6 = px.histogram(df2,x="Product_Category",color="Gender")

    graphJSON1 = json.dumps(plot1, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(plot2, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON3 = json.dumps(plot3, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON4 = json.dumps(plot4, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON5 = json.dumps(plot5, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON6 = json.dumps(plot6, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    return render_template('sales.html', graphJSON1=graphJSON1,graphJSON2=graphJSON2,graphJSON3=graphJSON3, graphJSON4  = graphJSON4,graphJSON5=graphJSON5,graphJSON6=graphJSON6)


#s14
@app.route("/topProrducts")
def topProducts():
    topProducts=pd.read_csv(r"static/dataset/section2/flipkart_top_products.csv")
    topProducts.rename(columns={'Title': 'name',
                   'Title_URL': 'product_link',
                   'Image': 'image_link',
                   '_3lwzlk': 'Rating',
                   '_30jeq3': 'price',
                   'rgwa7d' : 'Detail_1',#'Panel Type',
                   'rgwa7d2' : 'Detail_2',#'Screen Resolution Type',
                   'rgwa7d3' : 'Detail_3',#'Brightness',
                   'rgwa7d4' : 'Detail_4',#'Response Time'

                   },
          inplace=True, errors='raise')

    data = topProducts.filter(['name', 'product_link','Detail_1','Detail_2','Detail_3','Detail_4','image_link','price','Rating'])

    
    return render_template('topProducts.html', data = data)

if __name__ == "__main__":
    app.run(debug=True)

