export default function Footer() {
    return (
        <footer className="bg-gray-900 text-white py-8 mt-16">
            <div className="container mx-auto px-4">
                <div className="grid md:grid-cols-3 gap-8">
                    <div>
                        <h3 className="text-xl font-semibold mb-4">SolarAI</h3>
                        <p className="text-gray-400">
                            Advanced AI-powered solar panel dirt detection system for optimal energy production.
                        </p>
                    </div>
                    <div>
                        <h4 className="text-lg font-semibold mb-4">Quick Links</h4>
                        <ul className="space-y-2 text-gray-400">
                            <li><a href="#" className="hover:text-white transition-colors">Home</a></li>
                            <li><a href="#" className="hover:text-white transition-colors">API Documentation</a></li>
                            <li><a href="#" className="hover:text-white transition-colors">GitHub</a></li>
                        </ul>
                    </div>
                    <div>
                        <h4 className="text-lg font-semibold mb-4">Technology</h4>
                        <ul className="space-y-2 text-gray-400">
                            <li>PyTorch & Deep Learning</li>
                            <li>FastAPI Backend</li>
                            <li>Next.js Frontend</li>
                            <li>Computer Vision</li>
                        </ul>
                    </div>
                </div>
                <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
                    <p>&copy; 2024 SolarAI. Built with ❤️ for sustainable energy.</p>
                </div>
            </div>
        </footer>
    );
} 