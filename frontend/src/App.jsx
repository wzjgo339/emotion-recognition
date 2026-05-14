import { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import ResultDisplay from './components/ResultDisplay';
import ProbabilityChart from './components/ProbabilityChart';
import { predictEmotion } from './api';

export default function App() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleImageSelect = async (file) => {
    setImage(file);
    setError('');
    setResult(null);
    setLoading(true);

    try {
      const res = await predictEmotion(file);
      setResult(res.data);
    } catch (err) {
      const msg =
        err.response?.status === 413
          ? 'Image too large (max 10MB)'
          : err.response?.data?.detail || 'Recognition failed. Please try again.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="max-w-2xl mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800">
            Facial Expression Recognition
          </h1>
          <p className="text-gray-500 mt-1 text-sm">
            Upload a face image to recognize the emotion
          </p>
        </header>

        {/* Main card */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 space-y-6">
          <ImageUpload
            image={image}
            onImageSelect={handleImageSelect}
            disabled={loading}
          />

          {loading && (
            <div className="flex items-center justify-center gap-3 py-4">
              <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-gray-500">Analyzing...</span>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-600">
              {error}
            </div>
          )}

          {result && !loading && (
            <>
              <hr className="border-gray-100" />
              <ResultDisplay
                emotion={result.emotion}
                confidence={result.confidence}
                processingTime={result.processing_time_ms}
              />
              <ProbabilityChart
                probabilities={result.probabilities}
                predicted={result.emotion}
              />
            </>
          )}
        </div>

        <footer className="text-center text-xs text-gray-400 mt-6">
          Powered by PyTorch CNN + SelfAttention
        </footer>
      </div>
    </div>
  );
}
