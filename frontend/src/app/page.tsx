'use client';

import { useState } from 'react';
import ImageUpload from '@/components/ImageUpload';
import PredictionResult from '@/components/PredictionResult';
import LiveFeed from '@/components/LiveFeed';

import ModelSelector from '@/components/ModelSelector';

interface PredictionData {
  filename?: string;
  prediction?: string;
  class_name?: string;
  confidence: number;
  status?: string;
  efficiency_loss?: number;
}

export default function Home() {
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePrediction = async (result: PredictionData) => {
    setPrediction(result);
    setError(null);
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
    setPrediction(null);
  };

  const handleLoading = (isLoading: boolean) => {
    setLoading(isLoading);
  };

  return (
    <div className="p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header with Model Selector */}
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-8 pb-6 border-b border-gray-200">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 mb-2">
              Solar Panel Dirt Detection
            </h1>
            <p className="text-gray-600">
              Edge AI Monitoring & Stability Assessment
            </p>
          </div>
          <div className="mt-4 md:mt-0">
            <ModelSelector />
          </div>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column: Feed and Upload */}
          <div className="space-y-8">
            <LiveFeed />
            
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-semibold text-gray-800 mb-4">
                Upload Image
              </h2>
              
              <ImageUpload
                onPrediction={handlePrediction}
                onError={handleError}
                onLoading={handleLoading}
              />
              
              <div className="mt-6 pt-6 border-t border-gray-100">
                <h3 className="text-sm font-medium text-gray-500 mb-3 uppercase tracking-wider">Edge Controls</h3>
                <button
                  onClick={async () => {
                    handleLoading(true);
    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const res = await fetch(`${API_URL}/capture`, { method: 'POST' });
      const data = await res.json();
                    } catch {
                      handleError('Cannot connect to edge node');
                    } finally {
                      handleLoading(false);
                    }
                  }}
                  className="w-full flex items-center justify-center space-x-2 bg-purple-50 text-purple-700 px-4 py-3 rounded-lg border border-purple-200 hover:bg-purple-100 transition-colors"
                >
                  <span>📸</span>
                  <span className="font-semibold">Remote Pi Capture</span>
                </button>
              </div>
            </div>
          </div>

          {/* Right Column: Results */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Detection Results
            </h2>
            {loading && (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                <span className="ml-3 text-gray-600">Analyzing image...</span>
              </div>
            )}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-red-800">{error}</p>
              </div>
            )}
            {prediction && !loading && (
              <PredictionResult prediction={prediction} />
            )}
            {!prediction && !loading && !error && (
              <div className="text-center py-8 text-gray-500">
                <div className="text-6xl mb-4">📸</div>
                <p>Upload an image to get started</p>
              </div>
            )}
          </div>
        </div>

        {/* Features Section */}
        <div className="mt-16 bg-white rounded-lg shadow-lg p-8">
          <h2 className="text-3xl font-bold text-gray-900 text-center mb-8">
            How It Works
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-blue-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">📷</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">Upload Image</h3>
              <p className="text-gray-600">
                Upload a clear image of a solar panel from any angle
              </p>
            </div>
            <div className="text-center">
              <div className="bg-green-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🤖</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">AI Analysis</h3>
              <p className="text-gray-600">
                Our deep learning model analyzes the image for dirt detection
              </p>
            </div>
            <div className="text-center">
              <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">📊</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">Get Results</h3>
              <p className="text-gray-600">
                Receive instant results with confidence scores
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
