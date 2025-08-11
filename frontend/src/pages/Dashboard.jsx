const Dashboard = () => {
  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Data Driven Analysis of Political Sentiment across Social Media and News for Public Opinion Insights
        </h1>
        <p className="text-xl text-gray-600 max-w-4xl mx-auto mb-6">
          Advanced sentiment and emotion analysis for Nepali political content
          using machine learning models trained on 2022 Nepali election
          datasets.
        </p>
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 max-w-4xl mx-auto">
          <h2 className="text-lg font-semibold text-blue-900 mb-2">
            Dataset Information
          </h2>
          <p className="text-blue-800">
            This system is built upon datasets from the 2022 Nepali General
            Election, featuring political tweets and news articles that capture
            public sentiment. The models have been fine-tuned for Nepali
            language political discourse analysis.
          </p>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">
          Machine Learning Models
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-3">
              Sentiment Analysis Model
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">Model Type:</span>
                <span className="text-gray-900 font-medium">XLM-RoBERTa</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Classes:</span>
                <span className="text-gray-900 font-medium">
                  3 (Positive, Negative, Neutral)
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Training Data:</span>
                <span className="text-gray-900 font-medium">
                  2022 Election Tweets
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Language:</span>
                <span className="text-gray-900 font-medium">Nepali</span>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-3">
              Emotion Detection Model
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">Model Type:</span>
                <span className="text-gray-900 font-medium">XLM-RoBERTa</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Emotions:</span>
                <span className="text-gray-900 font-medium">6 Categories</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Training Data:</span>
                <span className="text-gray-900 font-medium">
                  2022 Election Content
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
