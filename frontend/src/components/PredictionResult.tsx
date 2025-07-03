interface PredictionResultProps {
    prediction: {
        filename: string;
        prediction: string;
        confidence: number;
        status: string;
    };
}

export default function PredictionResult({ prediction }: PredictionResultProps) {
    const isClean = prediction.prediction === 'clean';
    const confidencePercentage = (prediction.confidence * 100).toFixed(1);

    return (
        <div className="space-y-6">
            {/* Status Card */}
            <div className={`rounded-lg p-6 text-center ${isClean
                    ? 'bg-green-50 border border-green-200'
                    : 'bg-red-50 border border-red-200'
                }`}>
                <div className="text-6xl mb-4">
                    {isClean ? '✅' : '⚠️'}
                </div>
                <h3 className={`text-2xl font-bold mb-2 ${isClean ? 'text-green-800' : 'text-red-800'
                    }`}>
                    {isClean ? 'Clean Solar Panel' : 'Dirty Solar Panel'}
                </h3>
                <p className={`text-lg ${isClean ? 'text-green-700' : 'text-red-700'
                    }`}>
                    {isClean
                        ? 'Your solar panel appears to be clean and ready for optimal energy production!'
                        : 'Your solar panel may need cleaning to maintain optimal energy production.'
                    }
                </p>
            </div>

            {/* Confidence Score */}
            <div className="bg-gray-50 rounded-lg p-6">
                <h4 className="text-lg font-semibold text-gray-800 mb-4">Confidence Score</h4>
                <div className="space-y-3">
                    <div className="flex justify-between items-center">
                        <span className="text-gray-600">AI Confidence:</span>
                        <span className="font-semibold text-gray-800">{confidencePercentage}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                        <div
                            className={`h-3 rounded-full transition-all duration-500 ${confidencePercentage >= 80 ? 'bg-green-500' :
                                    confidencePercentage >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                                }`}
                            style={{ width: `${confidencePercentage}%` }}
                        ></div>
                    </div>
                    <div className="text-sm text-gray-500">
                        {confidencePercentage >= 80 ? 'High confidence' :
                            confidencePercentage >= 60 ? 'Medium confidence' : 'Low confidence'}
                    </div>
                </div>
            </div>

            {/* File Information */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-semibold text-blue-900 mb-2">File Information</h4>
                <div className="text-sm text-blue-800">
                    <p><strong>Filename:</strong> {prediction.filename}</p>
                    <p><strong>Status:</strong> {prediction.status}</p>
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