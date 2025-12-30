'use client';

import { useState, useEffect } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function ModelSelector() {
    const [currentModel, setCurrentModel] = useState('');
    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [loading, setLoading] = useState(false);

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
            }
        } catch (error) {
            console.error("Failed to fetch model config:", error);
        }
    };

    const handleModelChange = async (newModel: string) => {
        setLoading(true);
        try {
            const res = await fetch(`${API_URL}/config/model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: newModel }),
            });
            
            if (res.ok) {
                setCurrentModel(newModel);
            }
        } catch (error) {
            console.error("Error switching model:", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex items-center space-x-2 bg-gray-50 p-2 rounded-lg border border-gray-200">
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Active Model:
            </span>
            <div className="relative">
                <select
                    value={currentModel}
                    onChange={(e) => handleModelChange(e.target.value)}
                    disabled={loading}
                    className="appearance-none bg-white border border-gray-300 text-gray-700 text-sm rounded-md pl-3 pr-8 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 cursor-pointer"
                >
                    {availableModels.map((model) => (
                        <option key={model} value={model}>
                            {model}
                        </option>
                    ))}
                </select>
                {/* Custom chevron */}
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                    <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                        <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/>
                    </svg>
                </div>
            </div>
            {loading && (
                <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
            )}
        </div>
    );
}
