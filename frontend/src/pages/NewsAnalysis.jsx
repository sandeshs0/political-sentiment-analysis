import axios from "axios";
import {
  BarChart3,
  FileText,
  Filter,
  Globe,
  Newspaper,
  Search,
} from "lucide-react";
import { useState } from "react";
import { toast } from "react-hot-toast";
import LoadingSpinner from "../components/LoadingSpinner";
import PieChart from "../components/PieChart";
import StatsCard from "../components/StatsCard";

const NewsAnalysis = () => {
  const [formData, setFormData] = useState({
    sources: ["kathmandupost", "annapurna", "nagarik"],
    keywords: "",
    max_articles: 20,
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [sentimentFilter, setSentimentFilter] = useState("all"); // 'all', 'positive', 'negative', 'neutral'

  const emotionColors = {
    anger: "#ef4444",
    fear: "#8b5cf6",
    joy: "#f59e0b",
    love: "#ec4899",
    sadness: "#3b82f6",
    surprise: "#10b981",
  };

  const getFilteredArticles = () => {
    if (!results || !results.all_articles) return results?.top_articles || [];

    if (sentimentFilter === "all") {
      return results.all_articles
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 10);
    }

    const filtered = results.all_articles
      .filter(
        (article) =>
          article.sentiment.toLowerCase() === sentimentFilter.toLowerCase()
      )
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 10);

    return filtered;
  };

  const handleSourceChange = (source) => {
    setFormData((prev) => ({
      ...prev,
      sources: prev.sources.includes(source)
        ? prev.sources.filter((s) => s !== source)
        : [...prev.sources, source],
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (formData.sources.length === 0) {
      toast.error("Please select at least one news source");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post("/api/analyze-news", {
        sources: formData.sources,
        keywords: formData.keywords.trim() || null,
        max_articles: formData.max_articles,
      });

      if (response.data.success) {
        setResults(response.data.data);
        toast.success(`Analyzed ${response.data.data.total_articles} articles`);
      } else {
        toast.error(response.data.error || "Analysis failed");
      }
    } catch (error) {
      console.error("Analysis error:", error);
      toast.error(error.response?.data?.error || "Failed to analyze news");
    } finally {
      setLoading(false);
    }
  };

  const sourceLabels = {
    kathmandupost: "The Kathmandu Post",
    annapurna: "Annapurna Express",
    nagarik: "Nagarik News",
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-4 flex items-center justify-center gap-2">
          News Sentiment & Emotion Analysis
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Scrape and analyze sentiment and emotions in political news articles
          from major publications
        </p>
      </div>

      {/* Search Form */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* News Sources */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                News Sources
              </label>
              <div className="space-y-2">
                {Object.entries(sourceLabels).map(([value, label]) => (
                  <label key={value} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={formData.sources.includes(value)}
                      onChange={() => handleSourceChange(value)}
                      className="mr-3 h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                      disabled={loading}
                    />
                    <span className="text-sm text-gray-700">{label}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Keywords */}
            <div>
              <label
                htmlFor="keywords"
                className="block text-sm font-medium text-gray-700 mb-3"
              >
                Keywords (comma-separated)
              </label>
              <input
                type="text"
                id="keywords"
                value={formData.keywords}
                onChange={(e) =>
                  setFormData((prev) => ({ ...prev, keywords: e.target.value }))
                }
                placeholder="KP Oli, government, politics"
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={loading}
              />
              <p className="text-xs text-gray-500 mt-1">
                Optional: Filter articles by keywords
              </p>
            </div>

            {/* Max Articles */}
            <div>
              <label
                htmlFor="max_articles"
                className="block text-sm font-medium text-gray-700 mb-3"
              >
                Max Articles per Source
              </label>
              <select
                id="max_articles"
                value={formData.max_articles}
                onChange={(e) =>
                  setFormData((prev) => ({
                    ...prev,
                    max_articles: parseInt(e.target.value),
                  }))
                }
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={loading}
              >
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={30}>30</option>
                <option value={50}>50</option>
              </select>
            </div>
          </div>

          <div className="flex justify-center">
            <button
              type="submit"
              disabled={loading}
              className="px-8 py-3 bg-orange-600 hover:bg-orange-700 disabled:bg-orange-400 text-white font-medium rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
            >
              {loading ? (
                <p className="text-white font-medium">Analyzing...</p>
              ) : (
                <>
                  <Search className="h-4 w-4" />
                  Analyze News Articles
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="bg-white rounded-lg shadow-lg">
          <LoadingSpinner
            size="large"
            text="Scraping and analyzing news articles..."
          />
        </div>
      )}

      {/* Results */}
      {results && !loading && (
        <div className="space-y-8 fade-in">
          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <StatsCard
              title="Total Articles"
              value={results.total_articles}
              icon={FileText}
              color="primary"
            />
            <StatsCard
              title="Avg Sentiment Confidence"
              value={`${(
                results.analytics.avg_confidence.overall * 100
              ).toFixed(1)}%`}
              icon={BarChart3}
              color="success"
            />
            <StatsCard
              title="Most Common Sentiment"
              value={results.analytics.summary.most_common_sentiment}
              icon={Globe}
              color="warning"
            />
            <StatsCard
              title="Sources Analyzed"
              value={formData.sources.length}
              subtitle="News sources"
              icon={Newspaper}
              color="primary"
            />
          </div>
          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-1 gap-8">
            <PieChart
              data={results.sentiment_counts}
              title=" Sentiment Distribution"
              colors={["#22c55e", "#ef4444", "#6b7280"]}
            />
          </div>
          {results.source_analysis && (
            <div className="mt-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">
                {" "}
                Source-based Analysis
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {Object.entries(results.source_analysis).map(
                  ([source, analysis]) => (
                    <div
                      key={source}
                      className="bg-white rounded-lg shadow-lg p-6"
                    >
                      <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                        Media Outlet: {source}
                        <span className="text-sm text-gray-500">
                          ({analysis.total_articles} articles)
                        </span>
                      </h3>

                      {/* Sentiment Distribution for Source */}
                      <div className="mb-4">
                        <h4 className="text-sm font-medium text-gray-700 mb-2">
                          Sentiment Distribution
                        </h4>
                        <div className="grid grid-cols-3 gap-2 text-center">
                          {Object.entries(analysis.sentiment_distribution).map(
                            ([sentiment, percentage]) => (
                              <div
                                key={sentiment}
                                className={`p-2 rounded text-xs ${
                                  sentiment === "Positive"
                                    ? "bg-green-100 text-green-700"
                                    : sentiment === "Negative"
                                    ? "bg-red-100 text-red-700"
                                    : "bg-gray-100 text-gray-700"
                                }`}
                              >
                                <div className="font-bold">
                                  {(percentage * 100).toFixed(1)}%
                                </div>
                                <div>{sentiment}</div>
                              </div>
                            )
                          )}
                        </div>
                      </div>

                      {/* Emotion Distribution for Source */}
                      {analysis.emotion_distribution &&
                        Object.keys(analysis.emotion_distribution).length >
                          0 && (
                          <div className="mb-4">
                            <h4 className="text-sm font-medium text-gray-700 mb-2">
                              Top Emotions
                            </h4>
                            <div className="grid grid-cols-2 gap-1 text-xs">
                              {Object.entries(analysis.emotion_distribution)
                                .sort(([, a], [, b]) => b - a)
                                .slice(0, 4)
                                .map(([emotion, percentage]) => {
                                  const emojis = {
                                    anger: "😠",
                                    fear: "😨",
                                    joy: "😊",
                                    love: "❤️",
                                    sadness: "😢",
                                    surprise: "😮",
                                  };
                                  return (
                                    <div
                                      key={emotion}
                                      className="flex items-center gap-1 p-1 bg-gray-50 rounded"
                                    >
                                      <span>{emojis[emotion] || "😶"}</span>
                                      <span className="flex-1">{emotion}</span>
                                      <span className="font-bold">
                                        {(percentage * 100).toFixed(0)}%
                                      </span>
                                    </div>
                                  );
                                })}
                            </div>
                          </div>
                        )}

                      {/* Confidence Scores */}
                      <div className="text-xs text-gray-600 space-y-1">
                        <div>
                          Avg Sentiment Confidence:{" "}
                          {(analysis.avg_sentiment_confidence * 100).toFixed(1)}
                          %
                        </div>
                        {analysis.avg_emotion_confidence && (
                          <div>
                            Avg Emotion Confidence:{" "}
                            {(analysis.avg_emotion_confidence * 100).toFixed(1)}
                            %
                          </div>
                        )}
                      </div>

                      {/* Sample Articles */}
                      {analysis.top_positive.length > 0 && (
                        <div className="mt-3">
                          <h4 className="text-xs font-medium text-gray-700 mb-1">
                            Sample Positive
                          </h4>
                          <div className="text-xs text-gray-600">
                            <a
                              href={analysis.top_positive[0].url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:text-blue-800 line-clamp-2"
                            >
                              {analysis.top_positive[0].title.length > 60
                                ? analysis.top_positive[0].title.substring(
                                    0,
                                    60
                                  ) + "..."
                                : analysis.top_positive[0].title}
                            </a>
                          </div>
                        </div>
                      )}
                    </div>
                  )
                )}
              </div>
            </div>
          )}
          {/* Top Articles with Sentiment Meters and Filtering */}
          {results && (results.top_articles || results.all_articles) && (
            <div className="mt-8">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-800 mb-4 sm:mb-0">
                  📰 Top Articles by Confidence
                </h2>

                {/* Sentiment Filter */}
                <div className="flex items-center gap-2">
                  <Filter className="w-4 h-4 text-gray-600" />
                  <select
                    value={sentimentFilter}
                    onChange={(e) => setSentimentFilter(e.target.value)}
                    className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="all">All Sentiments</option>
                    <option value="positive">Positive Only</option>
                    <option value="negative">Negative Only</option>
                    <option value="neutral">Neutral Only</option>
                  </select>
                </div>
              </div>

              <div className="space-y-4">
                {getFilteredArticles().map((article, index) => (
                  <div
                    key={index}
                    className="bg-white rounded-lg shadow-lg p-6 border"
                  >
                    <div className="flex flex-col lg:flex-row lg:items-start gap-4">
                      {/* Article Content */}
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-gray-800 mb-2">
                          <a
                            href={article.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-600 hover:text-blue-800 hover:underline"
                          >
                            {article.title}
                          </a>
                        </h3>
                        <p className="text-sm text-gray-600 mb-2">
                          Author: {article.source_name} •{" "}
                          {article.author || "Unknown Author"}
                        </p>
                        {article.publication_date && (
                          <p className="text-xs text-gray-500 mb-3">
                            Date:{" "}
                            {new Date(
                              article.publication_date
                            ).toLocaleDateString()}
                          </p>
                        )}
                        <p className="text-gray-700 text-sm leading-relaxed">
                          {article.full_text.length > 300
                            ? article.full_text.substring(0, 300) + "..."
                            : article.full_text}
                        </p>
                      </div>

                      {/* Sentiment & Emotion Meters */}
                      <div className="lg:w-64 flex-shrink-0">
                        {/* Sentiment Meter */}
                        <div className="bg-gray-50 rounded-lg p-4 mb-3">
                          <h4 className="text-sm font-semibold text-gray-700 mb-2">
                            Sentiment
                          </h4>
                          <div className="flex items-center gap-2 mb-2">
                            <span
                              className={`px-2 py-1 rounded-full text-xs font-medium ${
                                article.sentiment === "Positive"
                                  ? "bg-green-100 text-green-800"
                                  : article.sentiment === "Negative"
                                  ? "bg-red-100 text-red-800"
                                  : "bg-gray-100 text-gray-800"
                              }`}
                            >
                              {article.sentiment}
                            </span>
                            <span className="text-xs text-gray-600">
                              {(article.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          {/* Sentiment Meter Bar */}
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full ${
                                article.sentiment === "Positive"
                                  ? "bg-green-500"
                                  : article.sentiment === "Negative"
                                  ? "bg-red-500"
                                  : "bg-gray-500"
                              }`}
                              style={{ width: `${article.confidence * 100}%` }}
                            />
                          </div>
                        </div>

                        {/* Emotion Meter */}
                        <div className="bg-gray-50 rounded-lg p-4">
                          <h4 className="text-sm font-semibold text-gray-700 mb-2">
                            Emotion
                          </h4>
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-lg">
                              {article.emotion === "anger"
                                ? "😠"
                                : article.emotion === "fear"
                                ? "😨"
                                : article.emotion === "joy"
                                ? "😊"
                                : article.emotion === "love"
                                ? "❤️"
                                : article.emotion === "sadness"
                                ? "😢"
                                : article.emotion === "surprise"
                                ? "😮"
                                : "😶"}
                            </span>
                            <span className="text-xs font-medium capitalize text-gray-700">
                              {article.emotion}
                            </span>
                            <span className="text-xs text-gray-600">
                              {article.emotion_confidence
                                ? (article.emotion_confidence * 100).toFixed(
                                    1
                                  ) + "%"
                                : "N/A"}
                            </span>
                          </div>
                          {/* Emotion Meter Bar */}
                          {article.emotion_confidence && (
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className="h-2 rounded-full bg-blue-500"
                                style={{
                                  width: `${article.emotion_confidence * 100}%`,
                                }}
                              />
                            </div>
                          )}
                        </div>

                        {/* Rank Badge */}
                        <div className="mt-3 text-center">
                          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                            #{index + 1} Most Confident
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Emotion Examples */}
          {results.top_emotions &&
            Object.keys(results.top_emotions).length > 0 && (
              <div className="mt-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-6">
                  {" "}
                  Emotion Examples from News
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {Object.entries(results.top_emotions).map(
                    ([emotion, articles]) => {
                      if (!articles || articles.length === 0) return null;

                      const emojis = {
                        anger: "😠",
                        fear: "😨",
                        joy: "😊",
                        love: "❤️",
                        sadness: "😢",
                        surprise: "😮",
                      };

                      return (
                        <div
                          key={emotion}
                          className="bg-white rounded-lg shadow-lg p-6"
                        >
                          <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
                            <span className="text-2xl">
                              {emojis[emotion] || "😶"}
                            </span>
                            {emotion.charAt(0).toUpperCase() + emotion.slice(1)}{" "}
                            Articles ({articles.length})
                          </h3>
                          <div className="space-y-3">
                            {articles.slice(0, 2).map((article, index) => (
                              <div
                                key={index}
                                className="bg-gray-50 p-3 rounded-lg"
                              >
                                <h4 className="text-sm font-medium text-gray-800 mb-1">
                                  <a
                                    href={article.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-blue-600 hover:text-blue-800"
                                  >
                                    {article.title.length > 80
                                      ? article.title.substring(0, 80) + "..."
                                      : article.title}
                                  </a>
                                </h4>
                                <p className="text-xs text-gray-600 mb-2">
                                  {article.source_name}
                                </p>
                                <div className="flex justify-between items-center">
                                  <span className="text-xs text-gray-500">
                                    Confidence:{" "}
                                    {(article.emotion_confidence * 100).toFixed(
                                      1
                                    )}
                                    %
                                  </span>
                                  <a
                                    href={article.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-xs text-blue-600 hover:text-blue-800"
                                  >
                                    Read
                                  </a>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    }
                  )}
                </div>
              </div>
            )}
        </div>
      )}
    </div>
  );
};

export default NewsAnalysis;
