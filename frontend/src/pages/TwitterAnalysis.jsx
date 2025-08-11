import axios from "axios";
import { BarChart3, MessageSquare, Search, TrendingUp } from "lucide-react";
import { useState } from "react";
import { toast } from "react-hot-toast";
import ContentCard from "../components/ContentCard";
import LoadingSpinner from "../components/LoadingSpinner";
import PieChart from "../components/PieChart";
import StatsCard from "../components/StatsCard";
import WordCloud from "../components/WordCloud";

const TwitterAnalysis = () => {
  const [keyword, setKeyword] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!keyword.trim()) {
      toast.error("Please enter a keyword to search");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post("/api/analyze-tweets", {
        keyword: keyword.trim(),
        max_results: 50,
      });

      if (response.data.success) {
        setResults(response.data.data);
        toast.success(`Analyzed ${response.data.data.total_tweets} tweets`);
      } else {
        toast.error(response.data.error || "Analysis failed");
      }
    } catch (error) {
      console.error("Analysis error:", error);
      toast.error(error.response?.data?.error || "Failed to analyze tweets");
    } finally {
      setLoading(false);
    }
  };

  const emotionColors = {
    anger: "#ef4444",
    fear: "#8b5cf6",
    joy: "#f59e0b",
    love: "#ec4899",
    sadness: "#3b82f6",
    surprise: "#10b981",
  };

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-4 flex items-center justify-center gap-2">
          Twitter Sentiment & Emotion Analysis
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Search for political keywords and analyze the sentiment and emotions
          in Nepali tweets
        </p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6">
        <form
          onSubmit={handleSubmit}
          className="flex flex-col sm:flex-row gap-4"
        >
          <div className="flex-1">
            <input
              type="text"
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
              placeholder="Enter keywords (e.g., KP Oli, सरकार, राजनीति)"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={loading}
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="px-8 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
          >
            {loading ? (
              <p> Analyzing...</p>
            ) : (
              <>
                <Search className="h-4 w-4" />
                Analyze Tweets
              </>
            )}
          </button>
        </form>
      </div>

      {loading && (
        <div className="bg-white rounded-lg shadow-lg">
          <LoadingSpinner
            size="large"
            text="Fetching and analyzing tweets..."
          />
        </div>
      )}

      {results && !loading && (
        <div className="space-y-8 fade-in">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <StatsCard
              title="Total Tweets"
              value={results.total_tweets}
              icon={MessageSquare}
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
              icon={TrendingUp}
              color="warning"
            />
            <StatsCard
              title="Confidence Range"
              value={results.analytics.summary.confidence_range}
              subtitle="Min - Max"
              icon={BarChart3}
              color="primary"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">
                {" "}
                Sentiment Distribution
              </h3>
              <div className="grid grid-cols-3 gap-3 text-center">
                {Object.entries(results.sentiment_counts).map(
                  ([sentiment, percentage]) => (
                    <div
                      key={sentiment}
                      className={`p-4 rounded-lg ${
                        sentiment === "Positive"
                          ? "bg-green-100"
                          : sentiment === "Negative"
                          ? "bg-red-100"
                          : "bg-gray-100"
                      }`}
                    >
                      <div
                        className={`text-2xl font-bold ${
                          sentiment === "Positive"
                            ? "text-green-700"
                            : sentiment === "Negative"
                            ? "text-red-700"
                            : "text-gray-700"
                        }`}
                      >
                        {(percentage * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-600">{sentiment}</div>
                    </div>
                  )
                )}
              </div>
            </div>

            {results.emotion_counts &&
              Object.keys(results.emotion_counts).length > 0 && (
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">
                    Emotion Distribution
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-center">
                    {Object.entries(results.emotion_counts).map(
                      ([emotion, percentage]) => {
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
                            className="p-3 rounded-lg bg-gray-50"
                          >
                            <div className="text-2xl mb-1">
                              {emojis[emotion] || "😶"}
                            </div>
                            <div className="text-lg font-bold text-gray-700">
                              {(percentage * 100).toFixed(1)}%
                            </div>
                            <div className="text-xs text-gray-600">
                              {emotion.charAt(0).toUpperCase() +
                                emotion.slice(1)}
                            </div>
                          </div>
                        );
                      }
                    )}
                  </div>
                </div>
              )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <PieChart
              data={results.sentiment_counts}
              title=" Sentiment Distribution"
              colors={["#22c55e", "#ef4444", "#6b7280"]}
            />
            {results.emotion_counts &&
              Object.keys(results.emotion_counts).length > 0 && (
                <PieChart
                  data={results.emotion_counts}
                  title=" Emotion Distribution"
                  colors={Object.keys(results.emotion_counts).map(
                    (emotion) => emotionColors[emotion] || "#6b7280"
                  )}
                />
              )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-green-50 border border-green-200 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-green-700 mb-4">
                Top Positive Tweets ({results.top_positive.length})
              </h3>
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {results.top_positive.map((tweet, index) => (
                  <ContentCard
                    key={index}
                    item={tweet}
                    type="tweet"
                    showEmotion={true}
                  />
                ))}
              </div>
            </div>

            <div className="bg-red-50 border border-red-200 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-red-700 mb-4">
                Top Negative Tweets ({results.top_negative.length})
              </h3>
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {results.top_negative.map((tweet, index) => (
                  <ContentCard
                    key={index}
                    item={tweet}
                    type="tweet"
                    showEmotion={true}
                  />
                ))}
              </div>
            </div>
          </div>

          {results.word_data && (
            <div className="mt-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">
                Word Clouds
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {results.word_data.overall.overall &&
                  results.word_data.overall.overall.length > 0 && (
                    <div className="bg-white rounded-lg shadow-lg p-6">
                      <h3 className="text-lg font-semibold text-gray-800 mb-3">
                        Overall
                      </h3>
                      <WordCloud
                        words={results.word_data.overall.overall}
                        sentiment="overall"
                        width={350}
                        height={180}
                      />
                    </div>
                  )}
                {results.word_data.overall.positive &&
                  results.word_data.overall.positive.length > 0 && (
                    <div className="bg-white rounded-lg shadow-lg p-6">
                      <h3 className="text-lg font-semibold text-green-800 mb-3">
                        Positive
                      </h3>
                      <WordCloud
                        words={results.word_data.overall.positive}
                        sentiment="positive"
                        width={350}
                        height={180}
                      />
                    </div>
                  )}
                {results.word_data.overall.negative &&
                  results.word_data.overall.negative.length > 0 && (
                    <div className="bg-white rounded-lg shadow-lg p-6">
                      <h3 className="text-lg font-semibold text-red-800 mb-3">
                        Negative
                      </h3>
                      <WordCloud
                        words={results.word_data.overall.negative}
                        sentiment="negative"
                        width={350}
                        height={180}
                      />
                    </div>
                  )}
                {results.word_data.overall.neutral &&
                  results.word_data.overall.neutral.length > 0 && (
                    <div className="bg-white rounded-lg shadow-lg p-6">
                      <h3 className="text-lg font-semibold text-gray-600 mb-3">
                        😐 Neutral
                      </h3>
                      <WordCloud
                        words={results.word_data.overall.neutral}
                        sentiment="neutral"
                        width={350}
                        height={180}
                      />
                    </div>
                  )}
              </div>

              {results.word_data.emotions &&
                Object.keys(results.word_data.emotions).length > 0 && (
                  <div className="mt-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">
                      Emotion Word Clouds
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {Object.entries(results.word_data.emotions).map(
                        ([emotion, words]) => {
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
                              className="bg-white rounded-lg shadow-lg p-4"
                            >
                              <h4 className="text-md font-semibold text-gray-700 mb-2 flex items-center gap-1">
                                <span>{emojis[emotion] || "😶"}</span>
                                {emotion.charAt(0).toUpperCase() +
                                  emotion.slice(1)}
                              </h4>
                              <WordCloud
                                words={words}
                                sentiment={emotion}
                                width={300}
                                height={150}
                              />
                            </div>
                          );
                        }
                      )}
                    </div>
                  </div>
                )}
            </div>
          )}

          {results.top_emotions &&
            Object.keys(results.top_emotions).length > 0 && (
              <div className="mt-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-6">
                  Emotion Examples
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {Object.entries(results.top_emotions).map(
                    ([emotion, tweets]) => {
                      if (!tweets || tweets.length === 0) return null;

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
                            Tweets ({tweets.length})
                          </h3>
                          <div className="space-y-3">
                            {tweets.slice(0, 3).map((tweet, index) => (
                              <div
                                key={index}
                                className="bg-gray-50 p-3 rounded-lg"
                              >
                                <p className="text-sm text-gray-700 mb-2">
                                  {tweet.text.length > 120
                                    ? tweet.text.substring(0, 120) + "..."
                                    : tweet.text}
                                </p>
                                <div className="flex justify-between items-center">
                                  <span className="text-xs text-gray-500">
                                    Confidence:{" "}
                                    {(tweet.emotion_confidence * 100).toFixed(
                                      1
                                    )}
                                    %
                                  </span>
                                  {tweet.tweet_url && (
                                    <a
                                      href={tweet.tweet_url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-xs text-blue-600 hover:text-blue-800"
                                    >
                                      View
                                    </a>
                                  )}
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

export default TwitterAnalysis;
