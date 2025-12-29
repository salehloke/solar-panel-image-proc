'use client';

import { useEffect, useState } from 'react';
import { 
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    LineChart, Line, Legend
} from 'recharts';
import { Activity, AlertTriangle, BarChart3, TrendingDown } from 'lucide-react';

interface AnalyticsData {
    total_detections: number;
    most_common_type: string;
    avg_efficiency_loss: number;
    class_distribution: Array<{ name: string, value: number }>;
    efficiency_trend: Array<{ time: string, loss: number }>;
    recent_history: Array<{ timestamp: string, class: string, loss: number }>;
}

export default function AnalyticsDashboard() {
    const [data, setData] = useState<AnalyticsData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch('http://localhost:8000/analytics');
                if (response.ok) {
                    const result = await response.json();
                    setData(result);
                }
            } catch (error) {
                console.error('Failed to fetch analytics:', error);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    if (loading) return <div className="p-8 text-center animate-pulse text-blue-600">Loading comprehensive analytics...</div>;
    if (!data) return <div className="p-8 text-center text-gray-500">No analytics data available</div>;

    return (
        <div className="space-y-8">
            {/* Top Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex items-center space-x-4">
                    <div className="bg-blue-100 p-3 rounded-lg text-blue-600">
                        <Activity size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-gray-500 font-medium uppercase tracking-wider">Total Detections</p>
                        <p className="text-2xl font-bold text-gray-900">{data.total_detections}</p>
                    </div>
                </div>
                
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex items-center space-x-4">
                    <div className="bg-orange-100 p-3 rounded-lg text-orange-600">
                        <AlertTriangle size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-gray-500 font-medium uppercase tracking-wider">Most Common Issue</p>
                        <p className="text-2xl font-bold text-gray-900">{data.most_common_type}</p>
                    </div>
                </div>

                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex items-center space-x-4">
                    <div className="bg-red-100 p-3 rounded-lg text-red-600">
                        <TrendingDown size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-gray-500 font-medium uppercase tracking-wider">Avg. Efficiency Loss</p>
                        <p className="text-2xl font-bold text-gray-900">{data.avg_efficiency_loss}%</p>
                    </div>
                </div>
            </div>

            {/* Charts Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Distribution Bar Chart */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                    <div className="flex items-center space-x-2 mb-6 text-gray-800">
                        <BarChart3 size={20} className="text-blue-500" />
                        <h3 className="font-bold text-lg">Detection Distribution</h3>
                    </div>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={data.class_distribution}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#6b7280', fontSize: 12}} />
                                <YAxis axisLine={false} tickLine={false} tick={{fill: '#6b7280', fontSize: 12}} />
                                <Tooltip 
                                    contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}
                                    cursor={{fill: '#f3f4f6'}}
                                />
                                <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Efficiency Trend Line Chart */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                    <div className="flex items-center space-x-2 mb-6 text-gray-800">
                        <Activity size={20} className="text-red-500" />
                        <h3 className="font-bold text-lg">Efficiency Loss Trend</h3>
                    </div>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={data.efficiency_trend}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{fill: '#6b7280', fontSize: 12}} />
                                <YAxis axisLine={false} tickLine={false} tick={{fill: '#6b7280', fontSize: 12}} unit="%" />
                                <Tooltip 
                                    contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}
                                />
                                <Legend />
                                <Line 
                                    type="monotone" 
                                    dataKey="loss" 
                                    name="Loss %"
                                    stroke="#ef4444" 
                                    strokeWidth={3} 
                                    dot={{r: 4, fill: '#ef4444'}}
                                    activeDot={{r: 6}}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Detailed History Table */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                <div className="bg-gray-50 px-6 py-4 border-b border-gray-100 flex justify-between items-center">
                    <h3 className="font-bold text-gray-800 text-lg">Recent Detections</h3>
                    <span className="text-xs font-semibold px-2 py-1 bg-gray-200 text-gray-600 rounded-full">LAST 20</span>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-gray-50 text-left">
                            <tr>
                                <th className="px-6 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                                <th className="px-6 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">Class</th>
                                <th className="px-6 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider text-right">Efficiency Loss</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                            {data.recent_history.map((item, idx) => (
                                <tr key={idx} className="hover:bg-gray-50 transition-colors">
                                    <td className="px-6 py-4 text-sm text-gray-600">
                                        {new Date(item.timestamp).toLocaleString()}
                                    </td>
                                    <td className="px-6 py-4">
                                        <span className={`text-xs font-bold px-2 py-1 rounded-full ${
                                            item.class === 'Clean' ? 'bg-green-100 text-green-700' :
                                            item.class === 'Dust' ? 'bg-yellow-100 text-yellow-700' :
                                            'bg-red-100 text-red-700'
                                        }`}>
                                            {item.class}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 text-sm text-right font-mono font-bold text-gray-900">
                                        {item.loss > 0 ? `-${item.loss}%` : '0%'}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
