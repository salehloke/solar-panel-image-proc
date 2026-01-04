'use client';

import AnalyticsDashboard from '@/components/AnalyticsDashboard';
import TrackerLogsTable from '@/components/TrackerLogsTable';

export default function HistoryPage() {
    return (
        <div className="p-6">
            <div className="max-w-6xl mx-auto space-y-8">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">Analysis History</h1>
                    <p className="text-gray-600">Track solar panel health and efficiency trends over time.</p>
                </div>
                
                {/* Historical Data Table */}
                <TrackerLogsTable />
                
                <div className="pt-8 border-t border-gray-200">
                    <h2 className="text-xl font-bold text-gray-900 mb-4">Analytics Overview</h2>
                    <AnalyticsDashboard />
                </div>
            </div>
        </div>
    );
}