export default function About() {
    return (
        <div className="p-6">
            <div className="max-w-4xl mx-auto">
                <div className="bg-white rounded-lg shadow-lg p-8">
                    <h1 className="text-3xl font-bold text-gray-900 mb-6">About SolarAI</h1>

                    <div className="prose prose-lg max-w-none">
                        <p className="text-gray-600 mb-6">
                            SolarAI is an advanced machine learning system designed to detect dirt and debris on solar panels
                            using computer vision and deep learning techniques. Our goal is to help solar panel owners and
                            maintenance teams identify when cleaning is needed to maintain optimal energy production.
                        </p>

                        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Technology Stack</h2>
                        <div className="grid md:grid-cols-2 gap-6 mb-8">
                            <div className="bg-blue-50 p-4 rounded-lg">
                                <h3 className="font-semibold text-blue-800 mb-2">Backend</h3>
                                <ul className="text-blue-700 space-y-1">
                                    <li>• FastAPI - High-performance web framework</li>
                                    <li>• PyTorch - Deep learning framework</li>
                                    <li>• ResNet18 - Pre-trained CNN architecture</li>
                                    <li>• Python 3.9+</li>
                                </ul>
                            </div>
                            <div className="bg-green-50 p-4 rounded-lg">
                                <h3 className="font-semibold text-green-800 mb-2">Frontend</h3>
                                <ul className="text-green-700 space-y-1">
                                    <li>• Next.js 14 - React framework</li>
                                    <li>• TypeScript - Type safety</li>
                                    <li>• Tailwind CSS - Styling</li>
                                    <li>• Responsive design</li>
                                </ul>
                            </div>
                        </div>

                        <h2 className="text-2xl font-semibold text-gray-800 mb-4">How It Works</h2>
                        <div className="space-y-4 mb-8">
                            <div className="flex items-start space-x-4">
                                <div className="bg-purple-100 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 mt-1">
                                    <span className="text-purple-800 font-bold text-sm">1</span>
                                </div>
                                <div>
                                    <h3 className="font-semibold text-gray-800">Image Upload</h3>
                                    <p className="text-gray-600">Users upload images of solar panels through our web interface</p>
                                </div>
                            </div>
                            <div className="flex items-start space-x-4">
                                <div className="bg-purple-100 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 mt-1">
                                    <span className="text-purple-800 font-bold text-sm">2</span>
                                </div>
                                <div>
                                    <h3 className="font-semibold text-gray-800">AI Processing</h3>
                                    <p className="text-gray-600">Our ResNet18 model analyzes the image to detect dirt patterns and debris</p>
                                </div>
                            </div>
                            <div className="flex items-start space-x-4">
                                <div className="bg-purple-100 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 mt-1">
                                    <span className="text-purple-800 font-bold text-sm">3</span>
                                </div>
                                <div>
                                    <h3 className="font-semibold text-gray-800">Results</h3>
                                    <p className="text-gray-600">Instant classification with confidence scores and cleaning recommendations</p>
                                </div>
                            </div>
                        </div>

                        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Model Performance</h2>
                        <div className="bg-gray-50 p-6 rounded-lg mb-8">
                            <div className="grid md:grid-cols-3 gap-6">
                                <div className="text-center">
                                    <div className="text-3xl font-bold text-blue-600 mb-2">95%+</div>
                                    <div className="text-gray-600">Accuracy</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-3xl font-bold text-green-600 mb-2">0.94</div>
                                    <div className="text-gray-600">F1 Score</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-3xl font-bold text-purple-600 mb-2">&lt;2s</div>
                                    <div className="text-gray-600">Response Time</div>
                                </div>
                            </div>
                        </div>

                        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Use Cases</h2>
                        <div className="grid md:grid-cols-2 gap-6 mb-8">
                            <div className="border border-gray-200 rounded-lg p-4">
                                <h3 className="font-semibold text-gray-800 mb-2">Solar Farms</h3>
                                <p className="text-gray-600">Monitor large-scale solar installations for maintenance scheduling</p>
                            </div>
                            <div className="border border-gray-200 rounded-lg p-4">
                                <h3 className="font-semibold text-gray-800 mb-2">Residential</h3>
                                <p className="text-gray-600">Homeowners can check their panels without professional inspection</p>
                            </div>
                            <div className="border border-gray-200 rounded-lg p-4">
                                <h3 className="font-semibold text-gray-800 mb-2">Maintenance Companies</h3>
                                <p className="text-gray-600">Prioritize cleaning services based on dirt detection results</p>
                            </div>
                            <div className="border border-gray-200 rounded-lg p-4">
                                <h3 className="font-semibold text-gray-800 mb-2">Research</h3>
                                <p className="text-gray-600">Academic and industry research on solar panel efficiency</p>
                            </div>
                        </div>

                        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                            <h3 className="font-semibold text-blue-800 mb-2">Get Started</h3>
                            <p className="text-blue-700 mb-4">
                                Ready to analyze your solar panels? Upload an image and get instant results!
                            </p>
                            <a
                                href="/"
                                className="inline-block bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                            >
                                Try It Now
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
} 