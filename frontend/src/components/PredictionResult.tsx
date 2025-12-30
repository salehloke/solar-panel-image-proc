interface PredictionResultProps {
    prediction: {
        filename?: string;
        prediction?: string;
        class_name?: string;
        confidence: number;
        status?: string;
        efficiency_loss?: number;
        image_url?: string;
        timestamp?: string;
        inference_time?: number;
        model_name?: string;
        // Model Benchmarks
        model_accuracy?: number;
        model_precision?: number;
        model_recall?: number;
        model_proc_time?: number;
    };
}

export default function PredictionResult({ prediction }: PredictionResultProps) {
    const label = prediction.class_name || prediction.prediction || 'Unknown';
    const isClean = label.toLowerCase() === 'clean';
    const confidenceNum = prediction.confidence * 100;
    const confidencePercentage = confidenceNum.toFixed(1);
    
    // Base URL for images served by the backend
    const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const fullImageUrl = prediction.image_url ? `${API_BASE_URL}${prediction.image_url}` : null;
    
    const classConfig: Record<string, { bg: string, border: string, text: string, icon: string, desc: string }> = {
        'clean': { 
            bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-800', icon: '✅',
            desc: 'Your solar panel appears to be clean and ready for optimal energy production!' 
        },
        'dust': { 
            bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-800', icon: '☁️',
            desc: 'A thin layer of dust detected. Efficiency might be slightly reduced.' 
        },
        'bird droppings': { 
            bg: 'bg-orange-50', border: 'border-orange-200', text: 'text-orange-800', icon: '💩',
            desc: 'Bird droppings detected. Localized hot spots may occur. Cleaning recommended.' 
        },
        'moss': { 
            bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-800', icon: '🌿',
            desc: 'Moss growth detected. Significant efficiency loss. Immediate cleaning required.' 
        }
    };

    const config = classConfig[label.toLowerCase()] || classConfig['clean'];

    return (
        <div className="space-y-6">
            {/* Status Card */}
            <div className={`rounded-lg p-6 text-center ${config.bg} border ${config.border}`}>
                <div className="text-6xl mb-4">
                    {config.icon}
                </div>
                <h3 className={`text-2xl font-bold mb-2 ${config.text}`}>
                    {label.charAt(0).toUpperCase() + label.slice(1)} Status
                </h3>
                <p className={`text-lg ${config.text}`}>
                    {config.desc}
                </p>
                {/* Show Model Used */}
                <p className="text-xs text-gray-500 mt-2 uppercase tracking-wide">
                    Model: {prediction.model_name || 'Unknown'}
                </p>
            </div>

            {/* Captured Image Display */}
            {fullImageUrl && (
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
                    <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
                        <span className="text-xs font-bold text-gray-500 uppercase">Captured Evidence</span>
                    </div>
                    <img 
                        src={fullImageUrl} 
                        alt="Captured solar panel" 
                        className="w-full h-auto object-cover max-h-64"
                    />
                </div>
            )}

            {/* Efficiency Loss Warning */}
            {prediction.efficiency_loss !== undefined && prediction.efficiency_loss > 0 && (
                <div className="bg-red-100 border-l-4 border-red-500 p-4 rounded">
                    <div className="flex items-center">
                        <div className="flex-shrink-0">
                            <span className="text-red-500 font-bold">⚠️</span>
                        </div>
                        <div className="ml-3">
                            <p className="text-sm text-red-700 font-bold">
                                Estimated Efficiency Loss: {prediction.efficiency_loss}%
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Analysis Details & Stability Benchmarks */}
            <div className="bg-gray-50 rounded-lg p-6">
                <div className="flex justify-between items-center mb-4">
                    <h4 className="text-lg font-semibold text-gray-800">Analysis Details</h4>
                    {prediction.timestamp && (
                        <span className="text-xs text-gray-400">
                            {new Date(prediction.timestamp).toLocaleTimeString()}
                        </span>
                    )}
                </div>
                
                <div className="space-y-6">
                    {/* Real-time Confidence */}
                    <div className="space-y-2">
                        <div className="flex justify-between items-center text-sm">
                            <span className="text-gray-600">AI Confidence (This Image):</span>
                            <span className="font-semibold text-gray-800">{confidencePercentage}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div
                                className={`h-2.5 rounded-full transition-all duration-500 ${confidenceNum >= 80 ? 'bg-green-500' :
                                        confidenceNum >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                                    }`}
                                style={{ width: `${confidencePercentage}%` }}
                            ></div>
                        </div>
                    </div>

                    {/* Stability Benchmarks (Model Level) */}
                    <div className="pt-4 border-t border-gray-200">
                        <h5 className="text-sm font-bold text-gray-700 mb-3 uppercase tracking-wider">Model Performance Benchmarks</h5>
                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-white p-3 rounded border border-gray-100 shadow-sm">
                                <p className="text-xs text-gray-500">Accuracy</p>
                                <p className="text-lg font-bold text-blue-600">
                                    {prediction.model_accuracy ? `${(prediction.model_accuracy * 100).toFixed(1)}%` : 'N/A'}
                                </p>
                            </div>
                            <div className="bg-white p-3 rounded border border-gray-100 shadow-sm">
                                <p className="text-xs text-gray-500">F1 Score</p>
                                <p className="text-lg font-bold text-purple-600">
                                    {prediction.model_accuracy ? (prediction.model_accuracy).toFixed(3) : 'N/A'}
                                </p>
                            </div>
                            <div className="bg-white p-3 rounded border border-gray-100 shadow-sm">
                                <p className="text-xs text-gray-500">Precision</p>
                                <p className="text-sm font-semibold text-gray-700">
                                    {prediction.model_precision ? (prediction.model_precision * 100).toFixed(1) + '%' : 'N/A'}
                                </p>
                            </div>
                            <div className="bg-white p-3 rounded border border-gray-100 shadow-sm">
                                <p className="text-xs text-gray-500">Recall</p>
                                <p className="text-sm font-semibold text-gray-700">
                                    {prediction.model_recall ? (prediction.model_recall * 100).toFixed(1) + '%' : 'N/A'}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Timing */}
                    <div className="flex flex-col space-y-2 pt-2">
                        <div className="flex justify-between text-xs text-gray-500">
                            <span>Real-time Inference:</span>
                            <span className="font-mono text-gray-700">
                                {prediction.inference_time ? `${(prediction.inference_time * 1000).toFixed(2)}ms` : 'N/A'}
                            </span>
                        </div>
                        <div className="flex justify-between text-xs text-gray-500">
                            <span>Benchmark Avg Proc Time:</span>
                            <span className="font-mono text-gray-700">
                                {prediction.model_proc_time ? `${prediction.model_proc_time.toFixed(4)}ms` : 'N/A'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Recommendations */}
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <h4 className="font-semibold text-yellow-900 mb-2">
                    {isClean ? '💡 Maintenance Tips' : '🧹 Cleaning Recommendations'}
                </h4>
                <div className="text-sm text-yellow-800 space-y-2">
                    {isClean ? (
                        <>
                            <p>• Continue regular monitoring to maintain optimal performance</p>
                            <p>• Schedule next inspection in 2-4 weeks</p>
                        </>
                    ) : (
                        <>
                            <p>• Clean with soft brushes and mild soap solution</p>
                            <p>• Avoid abrasive materials that could damage the surface</p>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
 