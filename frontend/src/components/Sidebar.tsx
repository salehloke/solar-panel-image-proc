'use client';

import { useState } from 'react';

interface SidebarProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function Sidebar({ isOpen, onClose }: SidebarProps) {
    const menuItems = [
        {
            id: 'home',
            label: 'Home',
            icon: '🏠',
            href: '/',
        },
        {
            id: 'upload',
            label: 'Upload Image',
            icon: '📷',
            href: '/upload',
        },
        {
            id: 'history',
            label: 'Analysis History',
            icon: '📊',
            href: '/history',
        },
        {
            id: 'settings',
            label: 'Settings',
            icon: '⚙️',
            href: '/settings',
        },
        {
            id: 'api',
            label: 'API Docs',
            icon: '📚',
            href: 'http://localhost:8000/docs',
            external: true,
        },
        {
            id: 'about',
            label: 'About',
            icon: 'ℹ️',
            href: '/about',
        },
    ];

    return (
        <>
            {/* Overlay */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
                    onClick={onClose}
                />
            )}

            {/* Sidebar */}
            <div className={`fixed top-0 left-0 h-full w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out z-50 ${isOpen ? 'translate-x-0' : '-translate-x-full'
                } lg:translate-x-0 lg:static lg:inset-0`}>

                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-200">
                    <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                            <span className="text-white font-bold text-sm">☀️</span>
                        </div>
                        <h2 className="text-xl font-bold text-gray-900">SolarAI</h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="lg:hidden p-2 rounded-lg hover:bg-gray-100 transition-colors"
                    >
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Navigation */}
                <nav className="p-4">
                    <div className="space-y-2">
                        {menuItems.map((item) => (
                            <a
                                key={item.id}
                                href={item.href}
                                target={item.external ? '_blank' : undefined}
                                rel={item.external ? 'noopener noreferrer' : undefined}
                                className="flex items-center space-x-3 px-4 py-3 rounded-lg text-gray-700 hover:bg-blue-50 hover:text-blue-700 transition-colors group"
                                onClick={!item.external ? onClose : undefined}
                            >
                                <span className="text-xl">{item.icon}</span>
                                <span className="font-medium">{item.label}</span>
                                {item.external && (
                                    <svg className="w-4 h-4 ml-auto opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                    </svg>
                                )}
                            </a>
                        ))}
                    </div>
                </nav>

                {/* Footer */}
                <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200">
                    <div className="text-center">
                        <p className="text-sm text-gray-500 mb-2">Powered by AI</p>
                        <div className="flex justify-center space-x-2">
                            <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">PyTorch</span>
                            <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">FastAPI</span>
                            <span className="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded">Next.js</span>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
} 