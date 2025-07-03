'use client';

import { useState, useRef } from 'react';

interface ImageUploadProps {
    onPrediction: (result: any) => void;
    onError: (error: string) => void;
    onLoading: (loading: boolean) => void;
}

export default function ImageUpload({ onPrediction, onError, onLoading }: ImageUploadProps) {
    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            // Validate file type
            if (!file.type.startsWith('image/')) {
                onError('Please select a valid image file');
                return;
            }

            // Validate file size (max 10MB)
            if (file.size > 10 * 1024 * 1024) {
                onError('Image size must be less than 10MB');
                return;
            }

            setSelectedImage(file);
            const url = URL.createObjectURL(file);
            setPreviewUrl(url);
            onError(''); // Clear any previous errors
        }
    };

    const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        const file = event.dataTransfer.files[0];
        if (file) {
            if (!file.type.startsWith('image/')) {
                onError('Please select a valid image file');
                return;
            }
            if (file.size > 10 * 1024 * 1024) {
                onError('Image size must be less than 10MB');
                return;
            }
            setSelectedImage(file);
            const url = URL.createObjectURL(file);
            setPreviewUrl(url);
            onError('');
        }
    };

    const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault();
    };

    const handleSubmit = async () => {
        if (!selectedImage) {
            onError('Please select an image first');
            return;
        }

        onLoading(true);
        onError('');

        try {
            const formData = new FormData();
            formData.append('file', selectedImage);

            const response = await fetch('http://localhost:8000/prediction/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            onPrediction(result);
        } catch (error) {
            console.error('Error uploading image:', error);
            onError('Failed to analyze image. Please try again.');
        } finally {
            onLoading(false);
        }
    };

    const handleReset = () => {
        setSelectedImage(null);
        setPreviewUrl(null);
        onError('');
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    return (
        <div className="space-y-6">
            {/* File Upload Area */}
            <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${previewUrl
                        ? 'border-green-300 bg-green-50'
                        : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
                    }`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
            >
                {previewUrl ? (
                    <div className="space-y-4">
                        <img
                            src={previewUrl}
                            alt="Preview"
                            className="max-w-full h-64 object-contain mx-auto rounded-lg"
                        />
                        <div className="space-y-2">
                            <p className="text-sm text-gray-600">
                                Selected: {selectedImage?.name}
                            </p>
                            <div className="flex justify-center space-x-2">
                                <button
                                    onClick={handleSubmit}
                                    className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                                >
                                    Analyze Image
                                </button>
                                <button
                                    onClick={handleReset}
                                    className="bg-gray-500 text-white px-4 py-2 rounded-lg hover:bg-gray-600 transition-colors"
                                >
                                    Reset
                                </button>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="space-y-4">
                        <div className="text-6xl mb-4">📷</div>
                        <p className="text-lg text-gray-600">
                            Drag and drop an image here, or click to select
                        </p>
                        <p className="text-sm text-gray-500">
                            Supports JPG, PNG, GIF (max 10MB)
                        </p>
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
                        >
                            Choose File
                        </button>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="image/*"
                            onChange={handleFileSelect}
                            className="hidden"
                        />
                    </div>
                )}
            </div>

            {/* Instructions */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-semibold text-blue-900 mb-2">📋 Tips for Best Results:</h4>
                <ul className="text-sm text-blue-800 space-y-1">
                    <li>• Ensure good lighting and clear visibility of the solar panel</li>
                    <li>• Capture the entire panel or a significant portion</li>
                    <li>• Avoid shadows or reflections that might obscure the surface</li>
                    <li>• Use high-resolution images for better accuracy</li>
                </ul>
            </div>
        </div>
    );
} 