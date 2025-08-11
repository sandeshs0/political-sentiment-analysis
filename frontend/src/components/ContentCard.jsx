import {
  AlertTriangle,
  ExternalLink,
  Frown,
  Heart,
  Meh,
  Smile,
  Zap,
} from "lucide-react";

const getEmotionIcon = (emotion) => {
  const icons = {
    anger: AlertTriangle,
    fear: Frown,
    joy: Smile,
    love: Heart,
    sadness: Frown,
    surprise: Zap,
  };
  return icons[emotion] || Meh;
};

const getEmotionEmoji = (emotion) => {
  const emojis = {
    anger: "😠",
    fear: "😨",
    joy: "😊",
    love: "❤️",
    sadness: "😢",
    surprise: "😮",
  };
  return emojis[emotion] || "😶";
};

const ContentCard = ({
  item,
  type = "tweet",
  showEmotion = true,
  maxLength = 150,
}) => {
  const getSentimentColor = (sentiment) => {
    switch (sentiment?.toLowerCase()) {
      case "positive":
        return "border-success-400 bg-success-50";
      case "negative":
        return "border-danger-400 bg-danger-50";
      default:
        return "border-gray-400 bg-gray-50";
    }
  };

  const getSentimentBadgeColor = (sentiment) => {
    switch (sentiment?.toLowerCase()) {
      case "positive":
        return "bg-success-100 text-success-800";
      case "negative":
        return "bg-danger-100 text-danger-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const displayText = type === "article" ? item.title : item.text;
  const truncatedText =
    displayText.length > maxLength
      ? displayText.substring(0, maxLength) + "..."
      : displayText;

  return (
    <div
      className={`bg-white p-4 rounded-lg border-l-4 ${getSentimentColor(
        item.sentiment
      )} card-hover`}
    >
      {type === "article" && (
        <>
          <h4 className="font-semibold text-gray-800 mb-2">
            <a
              href={item.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary-600 hover:text-primary-800 flex items-center gap-1"
            >
              {item.title}
              <ExternalLink className="h-3 w-3" />
            </a>
          </h4>
          <p className="text-sm text-gray-600 mb-2">{item.source_name}</p>
          <p className="text-sm text-gray-700 mb-3">
            {item.full_text?.substring(0, 150)}
            {item.full_text?.length > 150 && "..."}
          </p>
        </>
      )}

      {type === "tweet" && (
        <p className="text-sm text-gray-700 mb-3">{truncatedText}</p>
      )}

      <div className="flex flex-wrap justify-between items-center gap-2">
        <div className="flex flex-wrap gap-2">
          <span
            className={`px-2 py-1 rounded text-xs ${getSentimentBadgeColor(
              item.sentiment
            )}`}
          >
            {item.sentiment}: {(item.confidence * 100).toFixed(1)}%
          </span>

          {showEmotion && item.emotion && item.emotion_confidence && (
            <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs flex items-center gap-1">
              {getEmotionEmoji(item.emotion)}
              {item.emotion}: {(item.emotion_confidence * 100).toFixed(1)}%
            </span>
          )}
        </div>

        {item.tweet_url && (
          <a
            href={item.tweet_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary-600 hover:text-primary-800 text-xs font-medium flex items-center gap-1"
          >
            View Tweet
            <ExternalLink className="h-3 w-3" />
          </a>
        )}

        {type === "article" && item.url && (
          <a
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary-600 hover:text-primary-800 text-xs font-medium flex items-center gap-1"
          >
            Read Article
            <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </div>
    </div>
  );
};

export default ContentCard;
