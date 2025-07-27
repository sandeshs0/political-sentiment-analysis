from flask import Flask, render_template, request, jsonify, make_response
import tweepy
import re, emoji
import pandas as pd
from transformers import pipeline
import logging
import json
from collections import Counter
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from datetime import datetime

app = Flask(__name__)


logging.getLogger('werkzeug').setLevel(logging.ERROR)


BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAO0c3AEAAAAA5K1F00mhGPcO5mHbRTe87xJHBMU%3DyWh4fIbVLFXmBz6jtXBRKzjanpLL3FugcxHo3KuNCnJK7HQCeL"
client = tweepy.Client(bearer_token=BEARER_TOKEN)


model_path = "nepali-sentiment-model-xlmr"
sentiment_pipeline = pipeline("text-classification", model=model_path, tokenizer=model_path)
label_map = {"LABEL_0": "Neutral", "LABEL_1": "Positive", "LABEL_2": "Negative"}


with open("nepali_stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = set(f.read().splitlines())


class TwitterRateLimitError(Exception):
    pass

class TwitterServerError(Exception):
    pass

class TwitterAPIError(Exception):
    pass


def fetch_tweets(keyword, max_results=50):
    try:
        query = f"{keyword} lang:ne -is:retweet"
        tweets = client.search_recent_tweets(query=query, tweet_fields=["lang", "created_at"], max_results=max_results)
        return [tweet.text for tweet in tweets.data] if tweets.data else []
    except tweepy.TooManyRequests:
        raise TwitterRateLimitError("Rate limit exceeded")
    except tweepy.TwitterServerError:
        raise TwitterServerError("Twitter server error")
    except tweepy.Unauthorized:
        raise TwitterAPIError("Twitter API authentication failed")
    except tweepy.Forbidden:
        raise TwitterAPIError("Twitter API access forbidden")
    except Exception as e:
        raise TwitterAPIError(f"Twitter API error: {str(e)}")


def clean_tweet(text):
    if pd.isnull(text): return ""
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[\u200c\u200d\u200e\u200f\u202a-\u202e]", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"#\w+|@\w+|[a-zA-Z0-9०-९]", "", text)
    text = re.sub(r"[।!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]", "", text)
    words = text.strip().split()
    return " ".join([w for w in words if w not in stopwords and 2 <= len(w) <= 20])


def predict_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = label_map.get(result['label'], result['label'])
    return label, round(result['score'], 2)


def generate_analytics(df):
    """Generate comprehensive analytics from the tweet data"""
    analytics = {}
    
    
    sentiment_counts = df["sentiment"].value_counts(normalize=True).to_dict()
    analytics['sentiment_distribution'] = sentiment_counts
    
    
    analytics['avg_confidence'] = {
        'overall': round(df['confidence'].mean(), 2),
        'positive': round(df[df['sentiment'] == 'Positive']['confidence'].mean(), 2) if len(df[df['sentiment'] == 'Positive']) > 0 else 0,
        'negative': round(df[df['sentiment'] == 'Negative']['confidence'].mean(), 2) if len(df[df['sentiment'] == 'Negative']) > 0 else 0,
        'neutral': round(df[df['sentiment'] == 'Neutral']['confidence'].mean(), 2) if len(df[df['sentiment'] == 'Neutral']) > 0 else 0
    }
    
    
    analytics['confidence_bins'] = df['confidence'].tolist()
    
    
    df['tweet_length'] = df['text'].str.len()
    analytics['avg_tweet_length'] = {
        'overall': round(df['tweet_length'].mean(), 1),
        'positive': round(df[df['sentiment'] == 'Positive']['tweet_length'].mean(), 1) if len(df[df['sentiment'] == 'Positive']) > 0 else 0,
        'negative': round(df[df['sentiment'] == 'Negative']['tweet_length'].mean(), 1) if len(df[df['sentiment'] == 'Negative']) > 0 else 0,
        'neutral': round(df[df['sentiment'] == 'Neutral']['tweet_length'].mean(), 1) if len(df[df['sentiment'] == 'Neutral']) > 0 else 0
    }
    
    
    all_words = ' '.join(df['text']).split()
    word_freq = Counter(all_words)
    analytics['top_words'] = dict(word_freq.most_common(10))
    
    
    positive_words = ' '.join(df[df['sentiment'] == 'Positive']['text']).split() if len(df[df['sentiment'] == 'Positive']) > 0 else []
    negative_words = ' '.join(df[df['sentiment'] == 'Negative']['text']).split() if len(df[df['sentiment'] == 'Negative']) > 0 else []
    
    analytics['positive_words'] = dict(Counter(positive_words).most_common(5))
    analytics['negative_words'] = dict(Counter(negative_words).most_common(5))
    
    
    analytics['summary'] = {
        'total_tweets': len(df),
        'avg_confidence': analytics['avg_confidence']['overall'],
        'most_common_sentiment': df['sentiment'].mode().iloc[0] if len(df) > 0 else 'Unknown',
        'confidence_range': f"{df['confidence'].min():.2f} - {df['confidence'].max():.2f}"
    }
    
    return analytics

def generate_wordcloud_image(text_data, sentiment_type):
    """Generate word cloud image and return as base64 string"""
    if not text_data:
        return None
        
    try:
        
        combined_text = ' '.join(text_data)
        if len(combined_text.strip()) == 0:
            return None
            
        
        color_schemes = {
            'positive': ['#22c55e', '#16a34a', '#15803d', '#166534'],
            'negative': ['#ef4444', '#dc2626', '#b91c1c', '#991b1b'],
            'neutral': ['#6b7280', '#4b5563', '#374151', '#1f2937'],
            'overall': ['#3b82f6', '#1d4ed8', '#1e40af', '#1e3a8a']
        }
        
        colors = color_schemes.get(sentiment_type, color_schemes['overall'])
        
        
        wordcloud = WordCloud(
            width=400, 
            height=200, 
            background_color='white',
            colormap=plt.cm.Set3,
            max_words=50,
            relative_scaling=0.5,
            font_path=None
        ).generate(combined_text)
        
        
        img = io.BytesIO()
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()
        
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None


@app.errorhandler(TwitterRateLimitError)
def handle_rate_limit_error(e):
    return render_template("index.html", 
                         error_message="⚠️ Twitter API Rate Limit Exceeded! Please wait 15 minutes before trying again.")

@app.errorhandler(TwitterServerError)
def handle_server_error(e):
    return render_template("index.html", 
                         error_message=" Twitter server is currently unavailable. Please try again later.")

@app.errorhandler(TwitterAPIError)
def handle_api_error(e):
    return render_template("index.html", 
                         error_message=f" Twitter API Error: {str(e)}. Please check your connection and try again.")

@app.errorhandler(500)
def handle_internal_error(e):
    return render_template("index.html", 
                     error_message=" An internal server error occurred. Please try again.")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        keyword = request.form["keyword"]
        
        try:
            tweets = fetch_tweets(keyword, max_results=50)

            results = []
            for t in tweets:
                cleaned = clean_tweet(t)
                if cleaned:
                    label, score = predict_sentiment(cleaned)
                    results.append({"text": cleaned, "sentiment": label, "confidence": score})

            if not results:
                return render_template("index.html", 
                                     keyword=keyword,
                                     error_message="No tweets found for this keyword. Please try a different search term.")

            df = pd.DataFrame(results)

            sentiment_counts = df["sentiment"].value_counts(normalize=True).to_dict()
            top_positive = df[df["sentiment"] == "Positive"].sort_values("confidence", ascending=False).head(10)
            top_negative = df[df["sentiment"] == "Negative"].sort_values("confidence", ascending=False).head(10)

            
            analytics = generate_analytics(df)

            
            overall_wordcloud = generate_wordcloud_image(df['text'].tolist(), 'overall')
            positive_wordcloud = generate_wordcloud_image(df[df['sentiment'] == 'Positive']['text'].tolist(), 'positive')
            negative_wordcloud = generate_wordcloud_image(df[df['sentiment'] == 'Negative']['text'].tolist(), 'negative')

            return render_template("index.html",
                                   keyword=keyword,
                                   sentiment_counts=sentiment_counts,
                                   top_positive=top_positive.to_dict("records"),
                                   top_negative=top_negative.to_dict("records"),
                                   analytics=analytics,
                                   overall_wordcloud=overall_wordcloud,
                                   positive_wordcloud=positive_wordcloud,
                                   negative_wordcloud=negative_wordcloud)
        
        except TwitterRateLimitError:
            error_message = " Twitter API Rate Limit Exceeded! Please wait 15 minutes before trying again."
        except TwitterServerError:
            error_message = " Twitter server is currently unavailable. Please try again later."
        except TwitterAPIError as e:
            error_message = f" Twitter API Error: {str(e)}. Please check your connection and try again."
        except Exception as e:
            
            app.logger.error(f"Unexpected error: {str(e)}")
            error_message = " An unexpected error occurred. Please try again."

        return render_template("index.html", keyword=keyword, error_message=error_message)
    
    return render_template("index.html")

@app.route("/export/<keyword>")
def export_data(keyword):
    """Export search results as CSV"""
    try:
        
        tweets = fetch_tweets(keyword, max_results=100)  
        
        results = []
        for t in tweets:
            cleaned = clean_tweet(t)
            if cleaned:
                label, score = predict_sentiment(cleaned)
                results.append({
                    "original_text": t,
                    "cleaned_text": cleaned, 
                    "sentiment": label, 
                    "confidence": score,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        if not results:
            return "No data to export", 404
            
        df = pd.DataFrame(results)
        
        
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = f"attachment; filename=sentiment_analysis_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        response.headers["Content-type"] = "text/csv"
        
        return response
        
    except Exception as e:
        return f"Export failed: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
