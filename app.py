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

# Optional matplotlib import with error handling
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    print(f"Warning: matplotlib not available: {e}")
    MATPLOTLIB_AVAILABLE = False

# Optional wordcloud import
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception as e:
    print(f"Warning: wordcloud not available: {e}")
    WORDCLOUD_AVAILABLE = False

import numpy as np
from datetime import datetime
import os

# Import news scraper
try:
    from news_scraper import UnifiedNewsScraper
    NEWS_SCRAPER_AVAILABLE = True
except Exception as e:
    print(f"Warning: news scraper not available: {e}")
    NEWS_SCRAPER_AVAILABLE = False

app = Flask(__name__)


logging.getLogger('werkzeug').setLevel(logging.ERROR)


BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAPYN3gEAAAAA87wcMXWc%2BLcG4iVYqvh7sVy4xvc%3DYhik3JbYsYi5hrLx46C2jWDDSFE6VEVlcZaTlv7512wVmTPKPR"
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Model initialization with error handling
try:
    model_path = "nepali-sentiment-model-xlmr"
    sentiment_pipeline = pipeline("text-classification", model=model_path, tokenizer=model_path)
    label_map = {"LABEL_0": "Neutral", "LABEL_1": "Positive", "LABEL_2": "Negative"}
    print("✅ Sentiment model loaded successfully")
except Exception as e:
    print(f"Error loading sentiment model: {e}")
    sentiment_pipeline = None

# Emotion model initialization
try:
    emotion_model_path = "nepali-emotion-model-xlmr"
    emotion_pipeline = pipeline("text-classification", model=emotion_model_path, tokenizer=emotion_model_path)
    emotion_label_map = {
        "LABEL_0": "anger", "LABEL_1": "fear", "LABEL_2": "joy",
        "LABEL_3": "love", "LABEL_4": "sadness", "LABEL_5": "surprise"
    }
    print("✅ Emotion model loaded successfully")
except Exception as e:
    print(f"Error loading emotion model: {e}")
    emotion_pipeline = None

# Stopwords loading with error handling
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


def fetch_tweets(keyword, max_results=10):
    try:
        query = f"{keyword} lang:ne -is:retweet"
        tweets = client.search_recent_tweets(
            query=query, 
            tweet_fields=["lang", "created_at", "author_id", "public_metrics"], 
            max_results=max_results
        )
        
        if not tweets.data:
            return []
        
        # Return tweet objects with metadata
        tweet_objects = []
        for tweet in tweets.data:
            tweet_url = f"https://twitter.com/i/status/{tweet.id}"
            tweet_objects.append({
                'id': tweet.id,
                'text': tweet.text,
                'url': tweet_url,
                'created_at': tweet.created_at,
                'author_id': tweet.author_id
            })
        
        return tweet_objects
        
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
    
    # Sentiment analysis
    sentiment_counts = df["sentiment"].value_counts(normalize=True).to_dict()
    analytics['sentiment_distribution'] = sentiment_counts
    
    # Emotion analysis (if emotion column exists)
    if "emotion" in df.columns:
        emotion_counts = df["emotion"].value_counts(normalize=True).to_dict()
        analytics['emotion_distribution'] = emotion_counts
        
        # Emotion confidence stats
        analytics['avg_emotion_confidence'] = {
            'overall': round(df['emotion_confidence'].mean(), 2),
            'anger': round(df[df['emotion'] == 'anger']['emotion_confidence'].mean(), 2) if len(df[df['emotion'] == 'anger']) > 0 else 0,
            'fear': round(df[df['emotion'] == 'fear']['emotion_confidence'].mean(), 2) if len(df[df['emotion'] == 'fear']) > 0 else 0,
            'joy': round(df[df['emotion'] == 'joy']['emotion_confidence'].mean(), 2) if len(df[df['emotion'] == 'joy']) > 0 else 0,
            'love': round(df[df['emotion'] == 'love']['emotion_confidence'].mean(), 2) if len(df[df['emotion'] == 'love']) > 0 else 0,
            'sadness': round(df[df['emotion'] == 'sadness']['emotion_confidence'].mean(), 2) if len(df[df['emotion'] == 'sadness']) > 0 else 0,
            'surprise': round(df[df['emotion'] == 'surprise']['emotion_confidence'].mean(), 2) if len(df[df['emotion'] == 'surprise']) > 0 else 0
        }
    
    # Sentiment confidence stats
    analytics['avg_confidence'] = {
        'overall': round(df['confidence'].mean(), 2),
        'positive': round(df[df['sentiment'] == 'Positive']['confidence'].mean(), 2) if len(df[df['sentiment'] == 'Positive']) > 0 else 0,
        'negative': round(df[df['sentiment'] == 'Negative']['confidence'].mean(), 2) if len(df[df['sentiment'] == 'Negative']) > 0 else 0,
        'neutral': round(df[df['sentiment'] == 'Neutral']['confidence'].mean(), 2) if len(df[df['sentiment'] == 'Neutral']) > 0 else 0
    }
    
    # Confidence distribution
    analytics['confidence_bins'] = df['confidence'].tolist()
    
    # Tweet length analysis
    df['tweet_length'] = df['text'].str.len()
    analytics['avg_tweet_length'] = {
        'overall': round(df['tweet_length'].mean(), 1),
        'positive': round(df[df['sentiment'] == 'Positive']['tweet_length'].mean(), 1) if len(df[df['sentiment'] == 'Positive']) > 0 else 0,
        'negative': round(df[df['sentiment'] == 'Negative']['tweet_length'].mean(), 1) if len(df[df['sentiment'] == 'Negative']) > 0 else 0,
        'neutral': round(df[df['sentiment'] == 'Neutral']['tweet_length'].mean(), 1) if len(df[df['sentiment'] == 'Neutral']) > 0 else 0
    }
    
    # Word frequency analysis
    all_words = ' '.join(df['text']).split()
    word_freq = Counter(all_words)
    analytics['top_words'] = dict(word_freq.most_common(10))
    
    # Sentiment-specific word analysis
    positive_words = ' '.join(df[df['sentiment'] == 'Positive']['text']).split() if len(df[df['sentiment'] == 'Positive']) > 0 else []
    negative_words = ' '.join(df[df['sentiment'] == 'Negative']['text']).split() if len(df[df['sentiment'] == 'Negative']) > 0 else []
    
    analytics['positive_words'] = dict(Counter(positive_words).most_common(5))
    analytics['negative_words'] = dict(Counter(negative_words).most_common(5))
    
    # Summary statistics
    analytics['summary'] = {
        'total_tweets': len(df),
        'avg_confidence': analytics['avg_confidence']['overall'],
        'most_common_sentiment': df['sentiment'].mode().iloc[0] if len(df) > 0 else 'Unknown',
        'confidence_range': f"{df['confidence'].min():.2f} - {df['confidence'].max():.2f}"
    }
    
    # Add emotion summary if available
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
            
            # Get top emotion examples
            top_emotions = {}
            for emotion in ["anger", "fear", "joy", "love", "sadness", "surprise"]:
                emotion_df = df[df["emotion"] == emotion]
                if len(emotion_df) > 0:
                    top_emotions[emotion] = emotion_df.sort_values("emotion_confidence", ascending=False).head(5).to_dict("records")

            # Generate analytics
            analytics = generate_analytics(df)

            # Generate word clouds
            overall_wordcloud = generate_wordcloud_image(df['text'].tolist(), 'overall')
            positive_wordcloud = generate_wordcloud_image(df[df['sentiment'] == 'Positive']['text'].tolist(), 'positive')
            negative_wordcloud = generate_wordcloud_image(df[df['sentiment'] == 'Negative']['text'].tolist(), 'negative')
            
            # Generate emotion-specific word clouds
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
            
            app.logger.error(f"Unexpected error: {str(e)}")
            error_message = " An unexpected error occurred. Please try again."

        return render_template("index.html", keyword=keyword, error_message=error_message)
    
    return render_template("index.html")


@app.route("/news", methods=["GET", "POST"])
def news_analysis():
    if request.method == "POST":
        sources = request.form.getlist("sources")  # Multiple sources can be selected
        keywords = request.form.get("keywords", "").strip()
        max_articles = int(request.form.get("max_articles", 20))
        
        if not NEWS_SCRAPER_AVAILABLE:
            return render_template("news.html", 
                                 error_message="News scraper is not available. Please check if required dependencies are installed.")
        
        try:
            # Parse keywords
            keyword_list = [k.strip() for k in keywords.split(',') if k.strip()] if keywords else None
            
            # Initialize news scraper
            scraper = UnifiedNewsScraper(keywords=keyword_list)
            
            # Scrape articles
            if not sources:
                sources = ['kathmandupost', 'annapurna', 'nagarik']  # Default to all sources
                
            all_articles = scraper.scrape_sources(sources, max_articles)
            
            # Combine all articles for analysis
            combined_articles = []
            for source_articles in all_articles.values():
                for article in source_articles:
                    combined_articles.append(article.to_dict())
            
            if not combined_articles:
                return render_template("news.html", 
                                     sources=sources,
                                     keywords=keywords,
                                     error_message="No articles found. Try different sources or keywords.")
            
            # Analyze sentiment and emotion for each article
            results = []
            for article in combined_articles:
                # Combine title and content for analysis
                text_for_analysis = f"{article['title']} {article['full_text']}"
                cleaned = clean_tweet(text_for_analysis)  # Reuse cleaning function
                
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
            
            # Generate analytics similar to tweets
            sentiment_counts = df["sentiment"].value_counts(normalize=True).to_dict()
            emotion_counts = df["emotion"].value_counts(normalize=True).to_dict() if "emotion" in df.columns else {}
            
            # Get top articles by sentiment
            top_positive = df[df["sentiment"] == "Positive"].sort_values("confidence", ascending=False).head(5)
            top_negative = df[df["sentiment"] == "Negative"].sort_values("confidence", ascending=False).head(5)
            
            # Get top articles by emotion
            top_emotions = {}
            for emotion in ["anger", "fear", "joy", "love", "sadness", "surprise"]:
                emotion_df = df[df["emotion"] == emotion]
                if len(emotion_df) > 0:
                    top_emotions[emotion] = emotion_df.sort_values("emotion_confidence", ascending=False).head(3).to_dict("records")
            
            # Generate analytics
            analytics = generate_analytics(df)
            
            # Generate word clouds (optional)
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
        
        tweets = fetch_tweets(keyword, max_results=100)  
        
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
        print(f"❌ Missing required files/directories: {missing_items}")
        return False
    
    print("✅ All required files found")
    return True

def create_missing_files():
    """Create basic required files if they don't exist"""
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    
    # Create basic index.html if it doesn't exist
    if not os.path.exists("templates/index.html"):
        basic_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nepali Political Sentiment Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .error { color: red; background: #ffebee; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .form-group { margin: 20px 0; }
        input[type="text"] { padding: 10px; width: 300px; border: 1px solid #ddd; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        .results { margin-top: 30px; }
        .wordcloud { text-align: center; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏛️ Nepali Political Sentiment Analysis</h1>
        
        {% if error_message %}
            <div class="error">{{ error_message }}</div>
        {% endif %}
        
        <form method="POST">
            <div class="form-group">
                <label for="keyword">Enter Political Keyword (in Nepali):</label><br>
                <input type="text" name="keyword" id="keyword" placeholder="राजनीति, सरकार, नेता..." value="{{ keyword or '' }}" required>
                <button type="submit">🔍 Analyze Sentiment</button>
            </div>
        </form>
        
        {% if sentiment_counts %}
        <div class="results">
            <h2>📊 Analysis Results for "{{ keyword }}"</h2>
            
            <h3>Sentiment Distribution:</h3>
            {% for sentiment, percentage in sentiment_counts.items() %}
                <p><strong>{{ sentiment }}:</strong> {{ "%.1f"|format(percentage * 100) }}%</p>
            {% endfor %}
            
            {% if overall_wordcloud %}
            <div class="wordcloud">
                <h3>📝 Word Cloud</h3>
                <img src="data:image/png;base64,{{ overall_wordcloud }}" alt="Word Cloud">
            </div>
            {% endif %}
            
            {% if top_positive %}
            <h3>✅ Most Positive Tweets:</h3>
            <ul>
                {% for tweet in top_positive[:5] %}
                    <li>{{ tweet.text }} (Confidence: {{ tweet.confidence }})</li>
                {% endfor %}
            </ul>
            {% endif %}
            
            {% if top_negative %}
            <h3>❌ Most Negative Tweets:</h3>
            <ul>
                {% for tweet in top_negative[:5] %}
                    <li>{{ tweet.text }} (Confidence: {{ tweet.confidence }})</li>
                {% endfor %}
            </ul>
            {% endif %}
            
            <p><a href="/export/{{ keyword }}">📥 Download as CSV</a></p>
        </div>
        {% endif %}
    </div>
</body>
</html>'''
        with open("templates/index.html", "w", encoding="utf-8") as f:
            f.write(basic_html)
        print("✅ Created templates/index.html")
    
    # Create basic stopwords file if it doesn't exist
    if not os.path.exists("nepali_stopwords.txt"):
        basic_stopwords = """र
को
मा
छ
हो
गर्न
भन्न
हुन
ले
लाई
नि
त
पनि
अनि
यो
त्यो
एक
दुई
तीन
यस
त्यस
अब
फेरि
बाट
सम्म
भन्दा
जना
लागि
द्वारा
संग
प्रति
उनी
उनको
उनका
हामी
हाम्रो
तपाई
तपाईको
म
मेरो
तिमी
तिम्रो"""
        with open("nepali_stopwords.txt", "w", encoding="utf-8") as f:
            f.write(basic_stopwords)
        print("✅ Created nepali_stopwords.txt")

if __name__ == "__main__":
    print("🚀 Starting Nepali Political Sentiment Analysis App...")
    
    # Create missing files
    create_missing_files()
    
    # Validate required files
    if not validate_required_files():
        print("❌ Some required files are missing. Created basic versions.")
        print("⚠️  Note: You may need to add your trained model in 'nepali-sentiment-model-xlmr' directory")
    
    print("🌐 Starting Flask server...")
    app.run(debug=True, host="127.0.0.1", port=5000)
