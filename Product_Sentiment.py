import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import nltk
nltk.download('vader_lexicon')


df = pd.read_csv('flipkart_product.csv',encoding='ISO-8859-1')

reviews = []
for review in df['Review']:
    if isinstance(review, str):
        review = re.sub(r'[^a-zA-Z\s]', '', review)
        review = review.lower()
        reviews.append(review)

sid = SentimentIntensityAnalyzer()
sentiments = []
for review in reviews:
    sentiment = sid.polarity_scores(review)
    sentiments.append(sentiment)

positive = 0
negative = 0
neutral = 0
for sentiment in sentiments:
    if sentiment['compound'] > 0:
        positive += 1
    elif sentiment['compound'] < 0:
        negative += 1
    else:
        neutral += 1

labels = ['Positive', 'Negative', 'Neutral']
sizes = [positive, negative, neutral]
colors = ['yellowgreen', 'lightcoral', 'gold']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Product Sentiment Analysis')
plt.show()

words = ' '.join(reviews)
wordcloud = WordCloud(width = 800, height = 800, background_color ='white', min_font_size = 10).generate(words)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()














