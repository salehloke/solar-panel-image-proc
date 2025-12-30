'use client';

import { useState, useEffect } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Settings() {
    const [confidenceThreshold, setConfidenceThreshold] = useState(0.7);
    const [autoSave, setAutoSave] = useState(true);
    const [notifications, setNotifications] = useState(true);
    const [theme, setTheme] = useState('light');
    
    // Model Config State
    const [currentModel, setCurrentModel] = useState('');
    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [featuresUsed, setFeaturesUsed] = useState<string[]>([]);
    const [modelLoading, setModelLoading] = useState(false);
    const [modelMessage, setModelMessage] = useState('');

    useEffect(() => {
        fetchModelConfig();
    }, []);

    const fetchModelConfig = async () => {
        try {
            const res = await fetch(`${API_URL}/config/model`);
            if (res.ok) {
                const data = await res.json();
                setCurrentModel(data.current_model);
                setAvailableModels(data.available_models);
                setFeaturesUsed(data.features_used);
            }
        } catch (error) {
            console.error("Failed to fetch model config:", error);
        }
    };

    const handleModelChange = async (newModel: string) => {
        setModelLoading(true);
        setModelMessage('');
        try {
            const res = await fetch(`${API_URL}/config/model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: newModel }),
            });
            
            if (res.ok) {
                setCurrentModel(newModel);
                setModelMessage(`Successfully switched to ${newModel}`);
                fetchModelConfig(); // Refresh features used
            } else {
                setModelMessage('Failed to switch model');
            }
        } catch (error) {
            console.error("Error switching model:", error);
            setModelMessage('Error switching model');
        } finally {
            setModelLoading(false);
        }
    };

    return (
        <div className="p-6">
            <div className="max-w-4xl mx-auto">
                <div className="bg-white rounded-lg shadow-lg p-8">
                    <h1 className="text-3xl font-bold text-gray-900 mb-6">Settings</h1>

                    <div className="space-y-8">
                        
                        {/* Model Selection (New Section) */}
                        <div className="border-b border-gray-200 pb-6">
                            <h2 className="text-xl font-semibold text-gray-800 mb-4">Edge AI Model</h2>
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Active Classification Model
                                    </label>
                                    <div className="flex items-center space-x-4">
                                        <select
                                            value={currentModel}
                                            onChange={(e) => handleModelChange(e.target.value)}
                                            disabled={modelLoading}
                                            className="block w-full max-w-md px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
                                        >
                                            {availableModels.map((model) => (
                                                <option key={model} value={model}>
                                                    {model}
                                                </option>
                                            ))}
                                        </select>
                                        {modelLoading && <span className="text-sm text-blue-600 animate-pulse">Switching...</span>}
                                    </div>
                                    
                                    {modelMessage && (
                                        <p className={`text-sm mt-2 ${modelMessage.includes('Success') ? 'text-green-600' : 'text-red-600'}`}>
                                            {modelMessage}
                                        </p>
                                    )}

                                    <div className="mt-3">
                                        <p className="text-sm text-gray-600">
                                            Features used by current model: 
                                            <span className="font-mono ml-2 bg-gray-100 px-2 py-1 rounded">
                                                {featuresUsed.join(', ')}
                                            </span>
                                        </p>
                                        <p className="text-xs text-gray-500 mt-1">
                                            RF = Random Forest (Robust), SVM = Support Vector Machine (Fast).
                                            GLCM uses texture, HOG uses edges.
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Detection Settings */}
                        <div className="border-b border-gray-200 pb-6">
                            <h2 className="text-xl font-semibold text-gray-800 mb-4">Detection Settings</h2>

                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Confidence Threshold: {confidenceThreshold}
                                    </label>
                                    <input
                                        type="range"
                                        min="0.1"
                                        max="0.9"
                                        step="0.1"
                                        value={confidenceThreshold}
                                        onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                    />
                                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                                        <span>0.1 (More sensitive)</span>
                                        <span>0.9 (More strict)</span>
                                    </div>
                                    <p className="text-sm text-gray-600 mt-2">
                                        Adjust the minimum confidence level required for a prediction to be considered valid.
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* Application Settings */}
                        <div className="border-b border-gray-200 pb-6">
                            <h2 className="text-xl font-semibold text-gray-800 mb-4">Application Settings</h2>

                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <h3 className="text-sm font-medium text-gray-700">Auto-save Results</h3>
                                        <p className="text-sm text-gray-500">Automatically save analysis results to your history</p>
                                    </div>
                                    <label className="relative inline-flex items-center cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={autoSave}
                                            onChange={(e) => setAutoSave(e.target.checked)}
                                            className="sr-only peer"
                                        />
                                        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                    </label>
                                </div>

                                <div className="flex items-center justify-between">
                                    <div>
                                        <h3 className="text-sm font-medium text-gray-700">Notifications</h3>
                                        <p className="text-sm text-gray-500">Receive notifications for completed analyses</p>
                                    </div>
                                    <label className="relative inline-flex items-center cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={notifications}
                                            onChange={(e) => setNotifications(e.target.checked)}
                                            className="sr-only peer"
                                        />
                                        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                    </label>
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Theme
                                    </label>
                                    <select
                                        value={theme}
                                        onChange={(e) => setTheme(e.target.value)}
                                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                                    >
                                        <option value="light">Light</option>
                                        <option value="dark">Dark</option>
                                        <option value="auto">Auto (System)</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        {/* API Settings */}
                        <div className="border-b border-gray-200 pb-6">
                            <h2 className="text-xl font-semibold text-gray-800 mb-4">API Configuration</h2>

                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Backend URL
                                    </label>
                                    <input
                                        type="text"
                                        defaultValue="http://localhost:8000"
                                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                                        placeholder="Enter backend API URL"
                                    />
                                    <p className="text-sm text-gray-500 mt-1">
                                        The URL of your FastAPI backend server
                                    </p>
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        API Key (Optional)
                                    </label>
                                    <input
                                        type="password"
                                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                                        placeholder="Enter API key if required"
                                    />
                                    <p className="text-sm text-gray-500 mt-1">
                                        API key for authenticated requests (if configured)
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* Data Management */}
                        <div className="border-b border-gray-200 pb-6">
                            <h2 className="text-xl font-semibold text-gray-800 mb-4">Data Management</h2>

                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <h3 className="text-sm font-medium text-gray-700">Clear Analysis History</h3>
                                        <p className="text-sm text-gray-500">Remove all saved analysis results</p>
                                    </div>
                                    <button className="px-4 py-2 text-sm font-medium text-red-700 bg-red-100 border border-red-300 rounded-md hover:bg-red-200 focus:outline-none focus:ring-2 focus:ring-red-500">
                                        Clear History
                                    </button>
                                </div>

                                <div className="flex items-center justify-between">
                                    <div>
                                        <h3 className="text-sm font-medium text-gray-700">Export Data</h3>
                                        <p className="text-sm text-gray-500">Download your analysis history as CSV</p>
                                    </div>
                                    <button className="px-4 py-2 text-sm font-medium text-blue-700 bg-blue-100 border border-blue-300 rounded-md hover:bg-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-500">
                                        Export
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* Save Button */}
                        <div className="flex justify-end space-x-4">
                            <button className="px-6 py-2 text-sm font-medium text-gray-700 bg-gray-100 border border-gray-300 rounded-md hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500">
                                Reset to Defaults
                            </button>
                            <button className="px-6 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500">
                                Save Settings
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
} 