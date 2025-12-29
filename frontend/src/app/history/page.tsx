import AnalyticsDashboard from '@/components/AnalyticsDashboard';

export default function HistoryPage() {
    return (
        <div className="p-6">
            <div className="max-w-6xl mx-auto">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-900">Analysis History</h1>
                    <p className="text-gray-600">Track solar panel health and efficiency trends over time.</p>
                </div>
                
                <AnalyticsDashboard />
            </div>
        </div>
    );
}
