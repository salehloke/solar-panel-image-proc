'use client';

import { useState } from 'react';
import { Camera, CameraOff } from 'lucide-react';

export default function LiveFeed() {
    const [isStreaming, setIsStreaming] = useState(false);
    const [streamKey, setStreamKey] = useState(0);
    const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    const toggleStream = () => {
        setIsStreaming(!isStreaming);
        setStreamKey(prev => prev + 1); // Refresh stream on toggle
    };

    return (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
            <div className="bg-gray-50 px-6 py-4 border-b border-gray-100 flex justify-between items-center">
                <div className="flex items-center space-x-2">
                    <Camera size={20} className={isStreaming ? "text-red-500 animate-pulse" : "text-gray-400"} />
                    <h3 className="font-bold text-gray-800 text-lg">Live Edge Feed</h3>
                </div>
                <button 
                    onClick={toggleStream}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
                        isStreaming 
                        ? "bg-red-50 text-red-600 hover:bg-red-100" 
                        : "bg-blue-50 text-blue-600 hover:bg-blue-100"
                    }`}
                >
                    {isStreaming ? (
                        <>
                            <CameraOff size={16} />
                            <span>Stop Feed</span>
                        </>
                    ) : (
                        <>
                            <Camera size={16} />
                            <span>Start Feed</span>
                        </>
                    )}
                </button>
            </div>
            
            <div className="relative aspect-video bg-black flex items-center justify-center">
                {isStreaming ? (
                    <img 
                        src={`${API_BASE_URL}/stream?t=${streamKey}`} 
                        alt="Live Stream" 
                        className="w-full h-full object-contain"
                        onError={() => setIsStreaming(false)}
                    />
                ) : (
                    <div className="text-center text-gray-500 p-8">
                        <CameraOff size={48} className="mx-auto mb-4 opacity-20" />
                        <p>Feed is currently inactive</p>
                        <p className="text-xs mt-2 italic">Click &apos;Start Feed&apos; to begin real-time monitoring</p>
                    </div>
                )}
                
                {isStreaming && (
                    <div className="absolute top-4 right-4 bg-red-600 text-white text-xs font-bold px-2 py-1 rounded flex items-center space-x-1 shadow-lg">
                        <span className="w-2 h-2 bg-white rounded-full animate-ping"></span>
                        <span>LIVE</span>
                    </div>
                )}
            </div>
            
            <div className="p-4 bg-gray-50 text-xs text-gray-500 flex justify-between">
                <span>Node: solar-pi-01</span>
                <span>Format: MJPEG (320x240)</span>
            </div>
        </div>
    );
}
