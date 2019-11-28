import nltk
import plotly.express as px
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os
from textblob import TextBlob
pd.set_option('display.max_columns',10)
pd.set_option('display.max_rows',100)
pd.set_option('display.width',600)
#nltk.download()

def text_filter(text):
    return re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)


def text_filter2(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    wordsFiltered = []

    for w in words:
        if w not in stop_words:
            wordsFiltered.append(w)

    wordsFiltered = " ".join(wordsFiltered)
    return (wordsFiltered)


def sentiment_analysis(text):
    twitter_text = TextBlob(text)
    twitter_text = twitter_text.correct()
    result = twitter_text.sentiment.polarity
    return result


# Any results you write to the current directory are saved as output.
location_data = pd.read_csv('location_geocode.csv')
location_data=location_data.rename(columns={"name": "user_location"})


df_twitter = pd.read_csv('auspol2019.csv')
twitter_data = df_twitter[['created_at','full_text','user_location']]
twitter_data=twitter_data.sort_values(by='created_at')
twitter_data=twitter_data.reset_index(drop=True)
twitter_data=twitter_data.dropna()




#twitter_data=twitter_data.set_index(twitter_data['created_at'])
#twitter_data=sorted(twitter_data)



final_df=pd.DataFrame()
final_df=pd.merge(twitter_data,location_data,how='inner', on=['user_location'])
final_df=final_df.sort_values(by='created_at')
final_df=final_df.reset_index(drop=True)
#final_df['created_at']=sorted(final_df['created_at'])

print(twitter_data)
print(location_data)
print(final_df.head(1000))
print(final_df.columns)

final_df = final_df.iloc[len(final_df)-100000:len(final_df)]

print(final_df)

final_df['full_text']=final_df['full_text'].apply(text_filter)
final_df['full_text']=final_df['full_text'].apply(text_filter2)
final_df['Sentiment'] = final_df['full_text'].apply(sentiment_analysis)

final_df = final_df.astype('object')
final_df['Sentiment'] = final_df['Sentiment'].astype('float64')
print(final_df)

mapbox_access_token = 'pk.eyJ1IjoibGFraXRoYSIsImEiOiJjazNkeWk3amExZW9oM2NvMWYydWpzMjk5In0.sHirt2iGlDgPNn1fYMG68g'
px.set_mapbox_access_token(mapbox_access_token)
fig =px.scatter_mapbox(final_df, lat="lat", lon="long", color='Sentiment', text='user_location',
                        color_continuous_scale=px.colors.cyclical.IceFire, size_max=2, zoom=0.9)





fig.update_layout(

    height=800,
    width=1200,
    autosize=True,
)
fig.update_traces(marker=dict(size=12))
fig.update_yaxes(automargin=True)
fig.show()



