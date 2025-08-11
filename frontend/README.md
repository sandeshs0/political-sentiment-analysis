# Nepali Political Sentiment & Emotion Analysis Dashboard

A modern React-based dashboard for analyzing sentiment and emotions in Nepali political content from Twitter and news sources.

## Features

- **Real-time Twitter Analysis**: Search and analyze political tweets in Nepali
- **News Article Analysis**: Scrape and analyze articles from major Nepali news sources
- **Dual Analysis**: Both sentiment (Positive/Negative/Neutral) and emotion (6 emotions) detection
- **Interactive Visualizations**: Charts, graphs, and statistics
- **Professional UI**: Modern, responsive design with Tailwind CSS

## Tech Stack

### Frontend

- **React 18** - Modern React with hooks
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Chart.js** - Interactive charts and visualizations
- **React Router** - Client-side routing
- **Axios** - HTTP client for API calls
- **Lucide React** - Beautiful icons

### Backend

- **Flask** - Python web framework
- **Flask-CORS** - Cross-origin resource sharing
- **Transformers** - Hugging Face models for ML
- **Tweepy** - Twitter API integration
- **BeautifulSoup** - Web scraping for news articles
- **Pandas** - Data manipulation and analysis

## Getting Started

### Prerequisites

- Node.js 16+ and npm/yarn
- Python 3.8+
- Twitter API Bearer Token

### Frontend Setup

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

The frontend will run on `http://localhost:3000`

### Backend Setup

1. Navigate to the root directory and install Python dependencies:

```bash
pip install flask flask-cors transformers tweepy pandas numpy beautifulsoup4 feedparser requests
```

2. Download the ML models:

   - Place your Nepali sentiment model in `nepali-sentiment-model-xlmr/`
   - Place your Nepali emotion model in `nepali-emotion-model-xlmr/`

3. Set up Twitter API credentials in `app.py`

4. Start the Flask server:

```bash
python app.py
```

The backend will run on `http://localhost:5000`

## API Endpoints

### Health Check

```
GET /api/health
```

### Twitter Analysis

```
POST /api/analyze-tweets
Content-Type: application/json

{
  "keyword": "political keyword",
  "max_results": 50
}
```

### News Analysis

```
POST /api/analyze-news
Content-Type: application/json

{
  "sources": ["kathmandupost", "annapurna", "nagarik"],
  "keywords": "comma,separated,keywords",
  "max_articles": 20
}
```

## Development

### Frontend Development

```bash
cd frontend
npm run dev    # Start dev server
npm run build  # Build for production
npm run lint   # Run ESLint
```

### Backend Development

```bash
python app.py  # Start Flask server in debug mode
```

## Models

The application uses two fine-tuned XLM-RoBERTa models:

1. **Sentiment Analysis**: 3-class classification (Positive, Negative, Neutral)
2. **Emotion Detection**: 6-class classification (Anger, Fear, Joy, Love, Sadness, Surprise)

Both models are specifically trained for Nepali text analysis.

## News Sources

The application can scrape and analyze articles from:

- The Kathmandu Post
- Annapurna Express
- Nagarik News

## Features

### Twitter Analysis

- Real-time tweet fetching using Twitter API v2
- Sentiment and emotion analysis for each tweet
- Statistical breakdowns and visualizations
- Links to original tweets
- Top positive/negative tweets display
- Emotion-specific tweet examples

### News Analysis

- Multi-source news scraping
- Keyword-based filtering
- Full article content analysis
- Source-wise statistics
- Direct links to original articles
- Emotion-based article categorization

### Visualizations

- Interactive pie charts for distributions
- Statistical cards with key metrics
- Responsive grid layouts
- Color-coded sentiment indicators
- Emoji-based emotion representation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
