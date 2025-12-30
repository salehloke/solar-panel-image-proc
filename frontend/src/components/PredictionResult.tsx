interface PredictionResultProps {
    prediction: {
        filename?: string;
        prediction?: string; // Optional to support both backends
        class_name?: string; // Support for edge backend
        confidence: number;
        status?: string;
        efficiency_loss?: number;
        image_url?: string;
        timestamp?: string;
        inference_time?: number;
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
    
    // Mapping for colors and icons
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

            {/* Efficiency Loss Warning (If applicable) */}
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

            {/* Confidence Score */}
            <div className="bg-gray-50 rounded-lg p-6">
                <div className="flex justify-between items-center mb-4">
                    <h4 className="text-lg font-semibold text-gray-800">Analysis Details</h4>
                    {prediction.timestamp && (
                        <span className="text-xs text-gray-400">
                            {new Date(prediction.timestamp).toLocaleTimeString()}
                        </span>
                    )}
                </div>
                <div className="space-y-4">
                    <div className="space-y-2">
                        <div className="flex justify-between items-center text-sm">
                            <span className="text-gray-600">AI Confidence:</span>
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

                    <div className="flex justify-between text-xs border-t border-gray-100 pt-3">
                        <span className="text-gray-500">Inference Time:</span>
                        <span className="font-mono text-gray-700">{prediction.inference_time ? `${(prediction.inference_time * 1000).toFixed(2)}ms` : 'N/A'}</span>
                    </div>
                </div>
            </div>

            {/* File Information */}
            {(prediction.filename || prediction.status) && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h4 className="font-semibold text-blue-900 mb-2">File Information</h4>
                    <div className="text-sm text-blue-800">
                        {prediction.filename && <p><strong>Filename:</strong> {prediction.filename}</p>}
                        {prediction.status && <p><strong>Status:</strong> {prediction.status}</p>}
                    </div>
                </div>
            )}

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
                            <p>• Monitor energy production for any significant drops</p>
                        </>
                    ) : (
                        <>
                            <p>• Consider professional cleaning services</p>
                            <p>• Clean with soft brushes and mild soap solution</p>
                            <p>• Avoid abrasive materials that could damage the surface</p>
                            <p>• Schedule cleaning during early morning or evening hours</p>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
} 