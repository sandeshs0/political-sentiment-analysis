import { useEffect, useRef } from "react";

const WordCloud = ({
  words,
  width = 400,
  height = 200,
  sentiment = "overall",
}) => {
  const canvasRef = useRef(null);

  const colorSchemes = {
    positive: ["#22c55e", "#16a34a", "#15803d", "#166534"],
    negative: ["#ef4444", "#dc2626", "#b91c1c", "#991b1b"],
    neutral: ["#6b7280", "#4b5563", "#374151", "#1f2937"],
    overall: ["#3b82f6", "#1d4ed8", "#1e40af", "#1e3a8a"],
    anger: ["#ef4444", "#dc2626", "#b91c1c"],
    fear: ["#8b5cf6", "#7c3aed", "#6d28d9"],
    joy: ["#f59e0b", "#d97706", "#b45309"],
    love: ["#ec4899", "#db2777", "#be185d"],
    sadness: ["#3b82f6", "#2563eb", "#1d4ed8"],
    surprise: ["#10b981", "#059669", "#047857"],
  };

  useEffect(() => {
    if (!words || words.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    // Combine all text and get word frequencies
    const text = words.join(" ");
    const wordMap = {};

    // Simple word extraction and counting
    const cleanWords = text
      .toLowerCase()
      .replace(/[^\u0900-\u097F\u0980-\u09FF\s]/g, "") // Keep Devanagari and Bengali scripts
      .split(/\s+/)
      .filter((word) => word.length > 2); // Filter short words

    cleanWords.forEach((word) => {
      wordMap[word] = (wordMap[word] || 0) + 1;
    });

    // Get top words
    const sortedWords = Object.entries(wordMap)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 30);

    if (sortedWords.length === 0) return;

    const colors = colorSchemes[sentiment] || colorSchemes.overall;
    const maxCount = sortedWords[0][1];

    // Simple word cloud layout
    const words_to_draw = sortedWords.map(([word, count], index) => {
      const fontSize = Math.max(12, (count / maxCount) * 32);
      return {
        text: word,
        count,
        fontSize,
        color: colors[index % colors.length],
        x: Math.random() * (width - 100) + 50,
        y: Math.random() * (height - 50) + 30,
      };
    });

    // Draw words
    words_to_draw.forEach((wordObj) => {
      ctx.font = `${wordObj.fontSize}px Arial`;
      ctx.fillStyle = wordObj.color;
      ctx.textAlign = "center";
      ctx.fillText(wordObj.text, wordObj.x, wordObj.y);
    });
  }, [words, width, height, sentiment]);

  if (!words || words.length === 0) {
    return (
      <div
        className="flex items-center justify-center bg-gray-100 rounded-lg"
        style={{ width: `${width}px`, height: `${height}px` }}
      >
        <p className="text-gray-500 text-sm">No data available</p>
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="border rounded-lg bg-white"
      style={{ maxWidth: "100%", height: "auto" }}
    />
  );
};

export default WordCloud;
