from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
import tweepy
import re, emoji
import pandas as pd
from transformers import pipeline
import logging
import json
import time
from collections import Counter
import base64
import io

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    print(f"Warning: matplotlib not available: {e}")
    MATPLOTLIB_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception as e:
    print(f"Warning: wordcloud not available: {e}")
    WORDCLOUD_AVAILABLE = False

import numpy as np
from datetime import datetime
import os

try:
    from news_scraper import UnifiedNewsScraper
    NEWS_SCRAPER_AVAILABLE = True
except Exception as e:
    print(f"Warning: news scraper not available: {e}")
    NEWS_SCRAPER_AVAILABLE = False

app = Flask(__name__)
CORS(app) 




BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAKoW3gEAAAAAWKlD%2BVfWdQidS3XlWtuX45HuKBA%3DY0tePuqku5m896SsdpALFWyLAf2pyL6JlnhO6puBfHzbHcO3re"
client = tweepy.Client(bearer_token=BEARER_TOKEN)

try:
    model_path = "nepali-sentiment-model-xlmr"
    sentiment_pipeline = pipeline("text-classification", model=model_path, tokenizer=model_path)
    label_map = {"LABEL_0": "Neutral", "LABEL_1": "Positive", "LABEL_2": "Negative"}
    print(" Sentiment model loaded successfully")
except Exception as e:
    print(f"Error loading sentiment model: {e}")
    sentiment_pipeline = None

try:
    emotion_model_path = "nepali-emotion-model-xlmr"
    emotion_pipeline = pipeline("text-classification", model=emotion_model_path, tokenizer=emotion_model_path)
    emotion_label_map = {
        "LABEL_0": "anger", "LABEL_1": "fear", "LABEL_2": "joy",
        "LABEL_3": "love", "LABEL_4": "sadness", "LABEL_5": "surprise"
    }
    print(" Emotion model loaded successfully")
except Exception as e:
    print(f"Error loading emotion model: {e}")
    emotion_pipeline = None

try:
    with open("nepali_stopwords.txt", "r", encoding="utf-8") as f:
        stopwords = set(f.read().splitlines())
    print(" Stopwords loaded successfully")
except FileNotFoundError:
    print(" nepali_stopwords.txt not found. Using empty stopwords set.")
    stopwords = set()
except Exception as e:
    print(f" Error loading stopwords: {e}")
    stopwords = set()


class TwitterRateLimitError(Exception):
    pass

class TwitterServerError(Exception):
    pass

class TwitterAPIError(Exception):
    pass

def fetch_tweets(keyword, max_results=50):
    print(f" Starting Twitter API fetch for keyword: '{keyword}' with max_results: {max_results}")
    try:
        query = f"{keyword} lang:ne -is:retweet"
        print(f" Twitter API Query: {query}")
        
        tweets = client.search_recent_tweets(
            query=query, 
            tweet_fields=["lang", "created_at", "author_id", "public_metrics"], 
            max_results=max_results
        )
        
        if not tweets.data:
            print(" No tweets found in API response")
            return []
        
        print(f" Found {len(tweets.data)} tweets from Twitter API")
        
        
        tweet_objects = []
        for i, tweet in enumerate(tweets.data):
            tweet_url = f"https://twitter.com/i/status/{tweet.id}"
            tweet_objects.append({
                'id': tweet.id,
                'text': tweet.text,
                'url': tweet_url,
                'created_at': tweet.created_at,
                'author_id': tweet.author_id
            })
            print(f" Tweet {i+1}: ID={tweet.id}, Length={len(tweet.text)} chars")
        
        print(f"Successfully processed {len(tweet_objects)} tweet objects")
        return tweet_objects
        
    except tweepy.TooManyRequests:
        print(" Twitter API Rate Limit Error")
        raise TwitterRateLimitError("Rate limit exceeded")
    except tweepy.TwitterServerError:
        print(" Twitter Server Error")
        raise TwitterServerError("Twitter server error")
    except tweepy.Unauthorized:
        print(" Twitter API Authentication Failed")
        raise TwitterAPIError("Twitter API authentication failed")
    except tweepy.Forbidden:
        print(" Twitter API Access Forbidden")
        raise TwitterAPIError("Twitter API access forbidden")
    except Exception as e:
        print(f" Twitter API Exception: {str(e)}")
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
    if sentiment_pipeline is None:
        return "Unknown", 0.0
    
    try:
        result = sentiment_pipeline(text)[0]
        label = label_map.get(result['label'], result['label'])
        return label, round(result['score'], 2)
    except Exception as e:
        print(f"Error in sentiment prediction: {e}")
        return "Unknown", 0.0


def predict_emotion(text):
    if emotion_pipeline is None:
        return "Unknown", 0.0
    
    try:
        result = emotion_pipeline(text)[0]
        label = emotion_label_map.get(result['label'], result['label'])
        return label, round(result['score'], 2)
    except Exception as e:
        print(f"Error in emotion prediction: {e}")
        return "Unknown", 0.0


def generate_analytics(df):
    """Generate comprehensive analytics from the tweet data"""
    analytics = {}
    
    
    sentiment_counts = df["sentiment"].value_counts(normalize=True).to_dict()
    analytics['sentiment_distribution'] = sentiment_counts
    
    
    if "emotion" in df.columns:
        emotion_counts = df["emotion"].value_counts(normalize=True).to_dict()
        analytics['emotion_distribution'] = emotion_counts
        
        
        analytics['avg_emotion_confidence'] = {
            'overall': round(df['emotion_confidence'].mean(), 2),
            'anger': round(df[df['emotion'] == 'anger']['emotion_confidence'].mean(), 2) if len(df[df['emotion'] == 'anger']) > 0 else 0,
            'fear': round(df[df['emotion'] == 'fear']['emotion_confidence'].mean(), 2) if len(df[df['emotion'] == 'fear']) > 0 else 0,
            'joy': round(df[df['emotion'] == 'joy']['emotion_confidence'].mean(), 2) if len(df[df['emotion'] == 'joy']) > 0 else 0,
            'love': round(df[df['emotion'] == 'love']['emotion_confidence'].mean(), 2) if len(df[df['emotion'] == 'love']) > 0 else 0,
            'sadness': round(df[df['emotion'] == 'sadness']['emotion_confidence'].mean(), 2) if len(df[df['emotion'] == 'sadness']) > 0 else 0,
            'surprise': round(df[df['emotion'] == 'surprise']['emotion_confidence'].mean(), 2) if len(df[df['emotion'] == 'surprise']) > 0 else 0
        }
    
    
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
    
    
    if "emotion" in df.columns:
        analytics['summary']['most_common_emotion'] = df['emotion'].mode().iloc[0] if len(df) > 0 else 'Unknown'
        analytics['summary']['avg_emotion_confidence'] = analytics['avg_emotion_confidence']['overall']
    
    return analytics

def generate_wordcloud_image(text_data, sentiment_type):
    """Generate word cloud image and return as base64 string"""
    if not text_data or not WORDCLOUD_AVAILABLE or not MATPLOTLIB_AVAILABLE:
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


def generate_time_chart(time_data, chart_type='sentiment_by_hour'):
    if not MATPLOTLIB_AVAILABLE or not time_data:
        return None
        
    try:
        plt.figure(figsize=(12, 6))
        
        if chart_type == 'sentiment_by_hour':
            hours = list(range(24))
            sentiments = ['Positive', 'Negative', 'Neutral']
            colors = {'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#6b7280'}
            
            
            for sentiment in sentiments:
                if sentiment in time_data:
                    values = [time_data[sentiment].get(str(hour), 0) for hour in hours]
                    plt.plot(hours, values, marker='o', linewidth=2, 
                           label=sentiment, color=colors[sentiment])
            
            plt.xlabel('Hour of Day')
            plt.ylabel('Percentage')
            plt.title('Sentiment Distribution by Hour')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(range(0, 24, 2))
            
        elif chart_type == 'emotion_by_hour':
            hours = list(range(24))
            emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
            colors = {'anger': '#ef4444', 'fear': '#8b5cf6', 'joy': '#f59e0b', 
                     'love': '#ec4899', 'sadness': '#3b82f6', 'surprise': '#10b981'}
            
            for emotion in emotions:
                if emotion in time_data:
                    values = [time_data[emotion].get(str(hour), 0) for hour in hours]
                    plt.plot(hours, values, marker='o', linewidth=2, 
                           label=emotion.capitalize(), color=colors[emotion])
            
            plt.xlabel('Hour of Day')
            plt.ylabel('Percentage')
            plt.title('Emotion Distribution by Hour')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(range(0, 24, 2))
            
        elif chart_type == 'tweet_volume_by_hour':
            hours = list(time_data.keys())
            counts = list(time_data.values())
            
            plt.bar(hours, counts, color='#3b82f6', alpha=0.7)
            plt.xlabel('Hour of Day')
            plt.ylabel('Number of Tweets')
            plt.title('Tweet Volume by Hour')
            plt.grid(True, alpha=0.3)
            
        elif chart_type == 'sentiment_by_day':
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sentiments = ['Positive', 'Negative', 'Neutral']
            colors = {'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#6b7280'}
            
            x = np.arange(len(days))
            width = 0.25
            
            for i, sentiment in enumerate(sentiments):
                if sentiment in time_data:
                    values = [time_data[sentiment].get(day, 0) for day in days]
                    plt.bar(x + i*width, values, width, label=sentiment, 
                           color=colors[sentiment], alpha=0.7)
            
            plt.xlabel('Day of Week')
            plt.ylabel('Percentage')
            plt.title('Sentiment Distribution by Day of Week')
            plt.xticks(x + width, days, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()
        
    except Exception as e:
        print(f"Error generating time chart: {e}")
        return None


@app.errorhandler(TwitterRateLimitError)
def handle_rate_limit_error(e):
    return render_template("index.html", 
                         error_message=" Twitter API Rate Limit Exceeded! Please wait 15 minutes before trying again.")

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
            for tweet_obj in tweets:
                cleaned = clean_tweet(tweet_obj['text'])
                if cleaned:
                    sentiment_label, sentiment_score = predict_sentiment(cleaned)
                    emotion_label, emotion_score = predict_emotion(cleaned)
                    
                    results.append({
                        "text": cleaned,
                        "original_text": tweet_obj['text'],
                        "tweet_id": tweet_obj['id'],
                        "tweet_url": tweet_obj['url'],
                        "created_at": tweet_obj['created_at'],
                        "author_id": tweet_obj['author_id'],
                        "sentiment": sentiment_label, 
                        "confidence": sentiment_score,
                        "emotion": emotion_label,
                        "emotion_confidence": emotion_score
                    })

            if not results:
                return render_template("index.html", 
                                     keyword=keyword,
                                     error_message="No tweets found for this keyword. Please try a different search term.")

            df = pd.DataFrame(results)

            sentiment_counts = df["sentiment"].value_counts(normalize=True).to_dict()
            emotion_counts = df["emotion"].value_counts(normalize=True).to_dict() if "emotion" in df.columns else {}
            
            top_positive = df[df["sentiment"] == "Positive"].sort_values("confidence", ascending=False).head(10)
            top_negative = df[df["sentiment"] == "Negative"].sort_values("confidence", ascending=False).head(10)
            
            
            top_emotions = {}
            for emotion in ["anger", "fear", "joy", "love", "sadness", "surprise"]:
                emotion_df = df[df["emotion"] == emotion]
                if len(emotion_df) > 0:
                    top_emotions[emotion] = emotion_df.sort_values("emotion_confidence", ascending=False).head(5).to_dict("records")

            
            analytics = generate_analytics(df)

            
            overall_wordcloud = generate_wordcloud_image(df['text'].tolist(), 'overall')
            positive_wordcloud = generate_wordcloud_image(df[df['sentiment'] == 'Positive']['text'].tolist(), 'positive')
            negative_wordcloud = generate_wordcloud_image(df[df['sentiment'] == 'Negative']['text'].tolist(), 'negative')
            
            
            emotion_wordclouds = {}
            for emotion in ["anger", "fear", "joy", "love", "sadness", "surprise"]:
                emotion_df = df[df["emotion"] == emotion]
                if len(emotion_df) > 0:
                    emotion_wordclouds[emotion] = generate_wordcloud_image(emotion_df['text'].tolist(), emotion)

            return render_template("index.html",
                                   keyword=keyword,
                                   sentiment_counts=sentiment_counts,
                                   emotion_counts=emotion_counts,
                                   top_positive=top_positive.to_dict("records"),
                                   top_negative=top_negative.to_dict("records"),
                                   top_emotions=top_emotions,
                                   analytics=analytics,
                                   overall_wordcloud=overall_wordcloud,
                                   positive_wordcloud=positive_wordcloud,
                                   negative_wordcloud=negative_wordcloud,
                                   emotion_wordclouds=emotion_wordclouds)
        
        except TwitterRateLimitError:
            error_message = " Twitter API Rate Limit Exceeded! Please wait 15 minutes before trying again."
        except TwitterServerError:
            error_message = " Twitter server is currently unavailable. Please try again later."
        except TwitterAPIError as e:
            error_message = f" Twitter API Error: {str(e)}. Please check your connection and try again."
        except Exception as e:
            
            
            error_message = " An unexpected error occurred. Please try again."

        return render_template("index.html", keyword=keyword, error_message=error_message)
    
    return render_template("index.html")


@app.route("/news", methods=["GET", "POST"])
def news_analysis():
    if request.method == "POST":
        sources = request.form.getlist("sources")  
        keywords = request.form.get("keywords", "").strip()
        max_articles = int(request.form.get("max_articles", 20))
        
        if not NEWS_SCRAPER_AVAILABLE:
            return render_template("news.html", 
                                 error_message="News scraper is not available. Please check if required dependencies are installed.")
        try:
            keyword_list = [k.strip() for k in keywords.split(',') if k.strip()] if keywords else None
            scraper = UnifiedNewsScraper(keywords=keyword_list)
            if not sources:
                sources = ['kathmandupost', 'annapurna', 'nagarik']  
            all_articles = scraper.scrape_sources(sources, max_articles)
            combined_articles = []
            for source_articles in all_articles.values():
                for article in source_articles:
                    combined_articles.append(article.to_dict())
            if not combined_articles:
                return render_template("news.html", 
                                     sources=sources,
                                     keywords=keywords,
                                     error_message="No articles found. Try different sources or keywords.")
            results = []
            for article in combined_articles:
                
                text_for_analysis = f"{article['title']} {article['full_text']}"
                cleaned = clean_tweet(text_for_analysis)  
                
                if cleaned:
                    sentiment_label, sentiment_score = predict_sentiment(cleaned)
                    emotion_label, emotion_score = predict_emotion(cleaned)
                    
                    results.append({
                        "title": article['title'],
                        "full_text": article['full_text'],
                        "url": article['url'],
                        "source_name": article['source_name'],
                        "author": article['author'],
                        "publication_date": article['publication_date'],
                        "text": cleaned,
                        "sentiment": sentiment_label,
                        "confidence": sentiment_score,
                        "emotion": emotion_label,
                        "emotion_confidence": emotion_score
                    })
            if not results:
                return render_template("news.html",
                                     sources=sources,
                                     keywords=keywords,
                                     error_message="No articles could be analyzed. Please try different search terms.")
            df = pd.DataFrame(results)
            sentiment_counts = df["sentiment"].value_counts(normalize=True).to_dict()
            emotion_counts = df["emotion"].value_counts(normalize=True).to_dict() if "emotion" in df.columns else {}
            top_positive = df[df["sentiment"] == "Positive"].sort_values("confidence", ascending=False).head(5)
            top_negative = df[df["sentiment"] == "Negative"].sort_values("confidence", ascending=False).head(5)
            top_emotions = {}
            for emotion in ["anger", "fear", "joy", "love", "sadness", "surprise"]:
                emotion_df = df[df["emotion"] == emotion]
                if len(emotion_df) > 0:
                    top_emotions[emotion] = emotion_df.sort_values("emotion_confidence", ascending=False).head(3).to_dict("records")
            analytics = generate_analytics(df)
            overall_wordcloud = generate_wordcloud_image(df['text'].tolist(), 'overall')
            positive_wordcloud = generate_wordcloud_image(df[df['sentiment'] == 'Positive']['text'].tolist(), 'positive')
            negative_wordcloud = generate_wordcloud_image(df[df['sentiment'] == 'Negative']['text'].tolist(), 'negative')
            return render_template("news.html",
                                 sources=sources,
                                 keywords=keywords,
                                 sentiment_counts=sentiment_counts,
                                 emotion_counts=emotion_counts,
                                 top_positive=top_positive.to_dict("records"),
                                 top_negative=top_negative.to_dict("records"),
                                 top_emotions=top_emotions,
                                 analytics=analytics,
                                 overall_wordcloud=overall_wordcloud,
                                 positive_wordcloud=positive_wordcloud,
                                 negative_wordcloud=negative_wordcloud,
                                 total_articles=len(results))   
        except Exception as e:
            error_message = f"Error analyzing news: {str(e)}"
            return render_template("news.html", sources=sources, keywords=keywords, error_message=error_message)
    
    return render_template("news.html")

@app.route("/export/<keyword>")
def export_data(keyword):
    """Export search results as CSV"""
    try:
        
        tweets = fetch_tweets(keyword, max_results=50)  
        
        results = []
        for t in tweets:
            cleaned = clean_tweet(t)
            if cleaned:
                sentiment_label, sentiment_score = predict_sentiment(cleaned)
                emotion_label, emotion_score = predict_emotion(cleaned)
                
                results.append({
                    "original_text": t,
                    "cleaned_text": cleaned, 
                    "sentiment": sentiment_label, 
                    "sentiment_confidence": sentiment_score,
                    "emotion": emotion_label,
                    "emotion_confidence": emotion_score,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        if not results:
            return "No data to export", 404
            
        df = pd.DataFrame(results)
        
        
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = f"attachment; filename=sentiment_emotion_analysis_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        response.headers["Content-type"] = "text/csv"
        
        return response
        
    except Exception as e:
        return f"Export failed: {str(e)}", 500

def validate_required_files():
    """Check if all required files exist"""
    required_files = [
        "nepali_stopwords.txt",
        "templates/index.html"
    ]
    
    required_dirs = [
        "nepali-sentiment-model-xlmr",
        "nepali-emotion-model-xlmr",
        "templates"
    ]
    
    missing_items = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_items.append(file_path)
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_items.append(dir_path)
    
    if missing_items:
        print(f" Missing required files/directories: {missing_items}")
        return False
    
    print(" All required files found")
    return True


@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'message': 'API is running',
        'models': {
            'sentiment_available': sentiment_pipeline is not None,
            'emotion_available': emotion_pipeline is not None,
            'news_scraper_available': NEWS_SCRAPER_AVAILABLE
        }
    })

@app.route('/api/analyze-tweets', methods=['POST'])
def api_analyze_tweets():
    """API endpoint for Twitter analysis"""
    request_start_time = time.time()
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    
    print("="*60)
    print(f" NEW TWITTER API REQUEST at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Client IP: {client_ip}")
    
    try:
        
        
        data = request.get_json()
        if not data:
            print(" No JSON data provided in request")
            
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        keyword = data.get('keyword', '').strip()
        max_results = min(data.get('max_results', 50), 50)  

        print(f"Request Data: keyword='{keyword}', max_results={max_results}")
        
        
        if not keyword:
            print(" No keyword provided in request")
            
            return jsonify({'success': False, 'error': 'Keyword is required'}), 400
        
        
        print(f" API Request: Starting tweet fetch for keyword: '{keyword}' (max: {max_results})")
        
        tweets = fetch_tweets(keyword, max_results=max_results)
        print(f" API Response: Received {len(tweets)} tweets from fetch_tweets function")
        
        if not tweets:
            print(" No tweets returned, sending 404 response")
            return jsonify({
                'success': False, 
                'error': 'No tweets found for this keyword. Please try a different search term.'
            }), 404
        
        
        print(f" Starting analysis of {len(tweets)} tweets")
        results = []
        for i, tweet_obj in enumerate(tweets):
            cleaned = clean_tweet(tweet_obj['text'])
            if cleaned:
                print(f" Analyzing tweet {i+1}/{len(tweets)}: {cleaned[:50]}...")
                sentiment_label, sentiment_score = predict_sentiment(cleaned)
                emotion_label, emotion_score = predict_emotion(cleaned)
                
                results.append({
                    "text": cleaned,
                    "original_text": tweet_obj['text'],
                    "tweet_id": tweet_obj['id'],
                    "tweet_url": tweet_obj['url'],
                    "created_at": tweet_obj['created_at'].isoformat() if tweet_obj.get('created_at') else None,
                    "author_id": tweet_obj['author_id'],
                    "sentiment": sentiment_label, 
                    "confidence": float(sentiment_score),
                    "emotion": emotion_label,
                    "emotion_confidence": float(emotion_score) if emotion_score else None
                })
                print(f" Tweet {i+1} analyzed: Sentiment={sentiment_label}({sentiment_score:.3f}), Emotion={emotion_label}({emotion_score:.3f})")
            else:
                print(f" Tweet {i+1} skipped: Empty after cleaning")
        
        print(f" Analysis complete: {len(results)} tweets successfully analyzed out of {len(tweets)} fetched")
        
        if not results:
            print(" No tweets could be analyzed, sending 404 response")
            return jsonify({
                'success': False, 
                'error': 'No tweets could be analyzed. Please try different search terms.'
            }), 404
        
        df = pd.DataFrame(results)
        print(f" Created DataFrame with {len(df)} rows for analysis")
        
        
        sentiment_counts = df["sentiment"].value_counts(normalize=True).to_dict()
        emotion_counts = df["emotion"].value_counts(normalize=True).to_dict() if "emotion" in df.columns else {}
        print(f" Sentiment distribution: {sentiment_counts}")
        print(f"Emotion distribution: {emotion_counts}")
        
        
        top_positive = df[df["sentiment"] == "Positive"].sort_values("confidence", ascending=False).head(10)
        top_negative = df[df["sentiment"] == "Negative"].sort_values("confidence", ascending=False).head(10)
        print(f" Top positive tweets: {len(top_positive)}, Top negative tweets: {len(top_negative)}")
        
        
        top_emotions = {}
        for emotion in ["anger", "fear", "joy", "love", "sadness", "surprise"]:
            emotion_df = df[df["emotion"] == emotion]
            if len(emotion_df) > 0:
                top_emotions[emotion] = emotion_df.sort_values("emotion_confidence", ascending=False).head(5).to_dict("records")
        print(f"Top emotions found: {list(top_emotions.keys())}")
        
        
        print("Generating analytics...")
        analytics = generate_analytics(df)
        
        
        print(" Preparing word cloud data for frontend...")
        word_data = {
            'overall': df['text'].tolist(),
            'positive': df[df['sentiment'] == 'Positive']['text'].tolist(),
            'negative': df[df['sentiment'] == 'Negative']['text'].tolist(),
            'neutral': df[df['sentiment'] == 'Neutral']['text'].tolist(),
        }
        
        
        emotion_word_data = {}
        for emotion in ["anger", "fear", "joy", "love", "sadness", "surprise"]:
            emotion_df = df[df["emotion"] == emotion]
            if len(emotion_df) > 0:
                emotion_word_data[emotion] = emotion_df['text'].tolist()
        
        print(f"Prepared word data: overall={len(word_data['overall'])}, positive={len(word_data['positive'])}, negative={len(word_data['negative'])}")
        
        
        print("Generating time-based analysis...")
        time_analysis = {}
        time_charts = {}
        
        if 'created_at' in df.columns and df['created_at'].notna().any():
            
            df['created_at_parsed'] = pd.to_datetime(df['created_at'])
            df['hour'] = df['created_at_parsed'].dt.hour
            df['day_of_week'] = df['created_at_parsed'].dt.day_name()
            
            
            sentiment_by_hour = df.groupby(['hour', 'sentiment']).size().unstack(fill_value=0)
            sentiment_by_hour_pct = sentiment_by_hour.div(sentiment_by_hour.sum(axis=1), axis=0).fillna(0)
            
            
            emotion_by_hour = df.groupby(['hour', 'emotion']).size().unstack(fill_value=0)
            emotion_by_hour_pct = emotion_by_hour.div(emotion_by_hour.sum(axis=1), axis=0).fillna(0)
            
            
            sentiment_by_day = df.groupby(['day_of_week', 'sentiment']).size().unstack(fill_value=0)
            sentiment_by_day_pct = sentiment_by_day.div(sentiment_by_day.sum(axis=1), axis=0).fillna(0)
            
            time_analysis = {
                'sentiment_by_hour': sentiment_by_hour_pct.to_dict(),
                'emotion_by_hour': emotion_by_hour_pct.to_dict(),
                'sentiment_by_day': sentiment_by_day_pct.to_dict(),
                'tweet_count_by_hour': df['hour'].value_counts().sort_index().to_dict(),
                'tweet_count_by_day': df['day_of_week'].value_counts().to_dict()
            }
            
            
            print(" Generating time-based charts...")
            time_charts = {
                'sentiment_by_hour_chart': generate_time_chart(sentiment_by_hour_pct.to_dict(), 'sentiment_by_hour'),
                'emotion_by_hour_chart': generate_time_chart(emotion_by_hour_pct.to_dict(), 'emotion_by_hour'),
                'tweet_volume_chart': generate_time_chart(time_analysis['tweet_count_by_hour'], 'tweet_volume_by_hour'),
                'sentiment_by_day_chart': generate_time_chart(sentiment_by_day_pct.to_dict(), 'sentiment_by_day')
            }
        
        print(f" API Response ready: {len(results)} tweets analyzed successfully")
        
        request_duration = time.time() - request_start_time
        print(f" Total request processing time: {request_duration:.2f} seconds")
        print("="*60)
        
        return jsonify({
            'success': True,
            'data': {
                'keyword': keyword,
                'total_tweets': len(results),
                'sentiment_counts': sentiment_counts,
                'emotion_counts': emotion_counts,
                'top_positive': top_positive.to_dict("records"),
                'top_negative': top_negative.to_dict("records"),
                'top_emotions': top_emotions,
                'analytics': analytics,
                'word_data': {
                    'overall': word_data,
                    'emotions': emotion_word_data
                },
                'time_analysis': time_analysis,
                'time_charts': time_charts
            }
        })
        
    except TwitterRateLimitError:
        print("Twitter API Rate Limit Error caught in API endpoint")
        return jsonify({
            'success': False, 
            'error': 'Twitter API Rate Limit Exceeded! Please wait 15 minutes before trying again.'
        }), 429
    except TwitterServerError:
        print("witter Server Error caught in API endpoint")
        return jsonify({
            'success': False, 
            'error': 'Twitter server is currently unavailable. Please try again later.'
        }), 503
    except TwitterAPIError as e:
        print(f"witter API Error caught in API endpoint: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'Twitter API Error: {str(e)}. Please check your connection and try again.'
        }), 400
    except Exception as e:
        print(f"Unexpected error in API endpoint: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500

@app.route('/api/analyze-news', methods=['POST'])
def api_analyze_news():
    """API endpoint for News analysis"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        sources = data.get('sources', ['kathmandupost', 'annapurna', 'nagarik'])
        keywords = data.get('keywords', '').strip()
        max_articles = min(data.get('max_articles', 20), 50)  
        
        if not NEWS_SCRAPER_AVAILABLE:
            return jsonify({
                'success': False, 
                'error': 'News scraper is not available. Please check if required dependencies are installed.'
            }), 503
        
        if not sources:
            return jsonify({'success': False, 'error': 'At least one source must be selected'}), 400
        
        
        keyword_list = [k.strip() for k in keywords.split(',') if k.strip()] if keywords else None
        
        
        scraper = UnifiedNewsScraper(keywords=keyword_list)
        
        
        all_articles = scraper.scrape_sources(sources, max_articles)
        
        
        combined_articles = []
        for source_articles in all_articles.values():
            for article in source_articles:
                combined_articles.append(article.to_dict())
        
        if not combined_articles:
            return jsonify({
                'success': False, 
                'error': 'No articles found. Try different sources or keywords.'
            }), 404
        
        
        results = []
        for article in combined_articles:
            
            text_for_analysis = f"{article['title']} {article['full_text']}"
            cleaned = clean_tweet(text_for_analysis)  
            
            if cleaned:
                sentiment_label, sentiment_score = predict_sentiment(cleaned)
                emotion_label, emotion_score = predict_emotion(cleaned)
                
                results.append({
                    "title": article['title'],
                    "full_text": article['full_text'],
                    "url": article['url'],
                    "source_name": article['source_name'],
                    "author": article['author'],
                    "publication_date": article['publication_date'],
                    "text": cleaned,
                    "sentiment": sentiment_label,
                    "confidence": float(sentiment_score),
                    "emotion": emotion_label,
                    "emotion_confidence": float(emotion_score) if emotion_score else None
                })
        
        if not results:
            return jsonify({
                'success': False,
                'error': 'No articles could be analyzed. Please try different search terms.'
            }), 404
        
        df = pd.DataFrame(results)
        
        
        sentiment_counts = df["sentiment"].value_counts(normalize=True).to_dict()
        emotion_counts = df["emotion"].value_counts(normalize=True).to_dict() if "emotion" in df.columns else {}
        
        
        top_articles = df.nlargest(10, 'confidence').to_dict("records")
        
        
        top_emotions = {}
        for emotion in ["anger", "fear", "joy", "love", "sadness", "surprise"]:
            emotion_df = df[df["emotion"] == emotion]
            if len(emotion_df) > 0:
                top_emotions[emotion] = emotion_df.sort_values("emotion_confidence", ascending=False).head(3).to_dict("records")
        
        
        analytics = generate_analytics(df)
        
        
        print(" Generating source-based analysis...")
        source_analysis = {}
        for source in df['source_name'].unique():
            source_df = df[df['source_name'] == source]
            source_sentiment_counts = source_df['sentiment'].value_counts(normalize=True).to_dict()
            source_emotion_counts = source_df['emotion'].value_counts(normalize=True).to_dict()
            
            source_analysis[source] = {
                'total_articles': len(source_df),
                'sentiment_distribution': source_sentiment_counts,
                'emotion_distribution': source_emotion_counts,
                'avg_sentiment_confidence': source_df['confidence'].mean(),
                'avg_emotion_confidence': source_df['emotion_confidence'].mean() if 'emotion_confidence' in source_df.columns else None,
                'top_positive': source_df[source_df['sentiment'] == 'Positive'].sort_values('confidence', ascending=False).head(3).to_dict('records'),
                'top_negative': source_df[source_df['sentiment'] == 'Negative'].sort_values('confidence', ascending=False).head(3).to_dict('records')
            }
        
        
        print(" Preparing word cloud data for frontend...")
        word_data = {
            'overall': df['text'].tolist(),
            'positive': df[df['sentiment'] == 'Positive']['text'].tolist(),
            'negative': df[df['sentiment'] == 'Negative']['text'].tolist(),
            'neutral': df[df['sentiment'] == 'Neutral']['text'].tolist(),
        }
        
        
        emotion_word_data = {}
        for emotion in ["anger", "fear", "joy", "love", "sadness", "surprise"]:
            emotion_df = df[df["emotion"] == emotion]
            if len(emotion_df) > 0:
                emotion_word_data[emotion] = emotion_df['text'].tolist()
        
        
        source_word_data = {}
        for source in df['source_name'].unique():
            source_df = df[df['source_name'] == source]
            if len(source_df) > 0:
                source_word_data[source] = source_df['text'].tolist()
        
        
        all_articles_with_filters = df.to_dict("records")
        
        return jsonify({
            'success': True,
            'data': {
                'sources': sources,
                'keywords': keywords,
                'total_articles': len(results),
                'sentiment_counts': sentiment_counts,
                'emotion_counts': emotion_counts,
                'top_articles': top_articles,
                'all_articles': all_articles_with_filters,  
                'top_emotions': top_emotions,
                'analytics': analytics,
                'source_analysis': source_analysis,
                'word_data': {
                    'overall': word_data,
                    'emotions': emotion_word_data,
                    'sources': source_word_data
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500


if __name__ == "__main__":
    print(" Starting Nepali Political Sentiment Analysis App...")
    
    
    create_missing_files()
    
    
    if not validate_required_files():
        print("Some required files are missing. Created basic versions.")
        print("  Note: You may need to add your trained model in 'nepali-sentiment-model-xlmr' directory")
    
    print(" Starting Flask server...")
    app.run(debug=True, host="127.0.0.1", port=5000)
