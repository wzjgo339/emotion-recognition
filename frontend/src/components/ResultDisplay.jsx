const EMOTION_CONFIG = {
  angry: { emoji: '😠', label: 'Angry' },
  disgust: { emoji: '🤢', label: 'Disgust' },
  fear: { emoji: '😨', label: 'Fear' },
  happy: { emoji: '😊', label: 'Happy' },
  sad: { emoji: '😢', label: 'Sad' },
  surprise: { emoji: '😲', label: 'Surprise' },
  neutral: { emoji: '😐', label: 'Neutral' },
};

export default function ResultDisplay({ emotion, confidence, processingTime }) {
  const config = EMOTION_CONFIG[emotion] || EMOTION_CONFIG.neutral;

  return (
    <div className="text-center space-y-4">
      <div className="text-6xl">{config.emoji}</div>
      <div>
        <h2 className="text-2xl font-bold text-gray-800">
          {config.label}
          <span className="text-base font-normal text-gray-400 ml-2">({emotion})</span>
        </h2>
        <div className="mt-1 flex items-center justify-center gap-2">
          <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 rounded-full transition-all duration-500"
              style={{ width: `${(confidence * 100).toFixed(0)}%` }}
            />
          </div>
          <span className="text-sm font-medium text-gray-600">
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>
      {processingTime != null && (
        <p className="text-xs text-gray-400">
          Processed in {processingTime}ms
        </p>
      )}
    </div>
  );
}
