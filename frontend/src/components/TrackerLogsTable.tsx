'use client';

import { useState, useEffect, useMemo } from 'react';
import { 
    useReactTable, 
    getCoreRowModel, 
    getFilteredRowModel, 
    getPaginationRowModel,
    flexRender,
    createColumnHelper
} from '@tanstack/react-table';
import * as XLSX from 'xlsx';
import { Download, Search, ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';

interface DetectionRecord {
    class_name: string;
    confidence: number;
    efficiency_loss: number;
    inference_time: number;
    timestamp: string;
}

interface HistoryResponse {
    items: DetectionRecord[];
    total: number;
    page: number;
    size: number;
}

const columnHelper = createColumnHelper<DetectionRecord>();

export default function TrackerLogsTable() {
    const [data, setData] = useState<DetectionRecord[]>([]);
    const [loading, setLoading] = useState(true);
    const [globalFilter, setGlobalFilter] = useState('');
    const [pagination, setPagination] = useState({
        pageIndex: 0,
        pageSize: 10,
    });
    const [totalRows, setTotalRows] = useState(0);

    const fetchData = async (page: number, size: number) => {
        setLoading(true);
        try {
            const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
            const res = await fetch(`${API_URL}/detections/history?page=${page + 1}&size=${size}`);
            if (res.ok) {
                const result: HistoryResponse = await res.json();
                setData(result.items);
                setTotalRows(result.total);
            }
        } catch (error) {
            console.error("Failed to fetch history", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData(pagination.pageIndex, pagination.pageSize);
    }, [pagination.pageIndex, pagination.pageSize]);

    const columns = useMemo(() => [
        columnHelper.accessor('timestamp', {
            header: 'Timestamp',
            cell: info => new Date(info.getValue()).toLocaleString(),
        }),
        columnHelper.accessor('class_name', {
            header: 'Defect Type',
            cell: info => (
                <span className={`px-2 py-1 rounded-full text-xs font-bold ${
                    info.getValue() === 'Clean' ? 'bg-green-100 text-green-700' :
                    info.getValue() === 'Dust' ? 'bg-yellow-100 text-yellow-700' :
                    'bg-red-100 text-red-700'
                }`}>
                    {info.getValue()}
                </span>
            ),
        }),
        columnHelper.accessor('confidence', {
            header: 'Confidence',
            cell: info => `${(info.getValue() * 100).toFixed(1)}%`,
        }),
        columnHelper.accessor('efficiency_loss', {
            header: 'Efficiency Loss',
            cell: info => info.getValue() > 0 ? `-${info.getValue()}%` : '0%',
        }),
        columnHelper.accessor('inference_time', {
            header: 'Proc. Time',
            cell: info => `${(info.getValue() * 1000).toFixed(0)}ms`,
        }),
    ], []);

    const table = useReactTable({
        data,
        columns,
        pageCount: Math.ceil(totalRows / pagination.pageSize),
        state: {
            pagination,
            globalFilter,
        },
        onPaginationChange: setPagination,
        onGlobalFilterChange: setGlobalFilter,
        getCoreRowModel: getCoreRowModel(),
        getFilteredRowModel: getFilteredRowModel(), // Client-side filtering for currently loaded page (MVP)
        manualPagination: true, // Server-side pagination
    });

    const [exporting, setExporting] = useState(false);

    const exportToExcel = async () => {
        setExporting(true);
        try {
            const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
            // Fetch ALL data (large limit)
            const res = await fetch(`${API_URL}/detections/history?page=1&size=100000`); 
            if (!res.ok) throw new Error("Failed to fetch export data");
            
            const result: HistoryResponse = await res.json();
            
            const ws = XLSX.utils.json_to_sheet(result.items.map(row => ({
                Timestamp: new Date(row.timestamp).toLocaleString(),
                "Defect Type": row.class_name,
                Confidence: `${(row.confidence * 100).toFixed(1)}%`,
                "Efficiency Loss": `${row.efficiency_loss}%`,
                "Processing Time": `${row.inference_time}s`
            })));
            const wb = XLSX.utils.book_new();
            XLSX.utils.book_append_sheet(wb, ws, "TrackerLogs");
            XLSX.writeFile(wb, `SolarAI_Logs_ALL_${new Date().toISOString().split('T')[0]}.xlsx`);
        } catch (error) {
            console.error("Export failed", error);
            alert("Export failed. Please try again.");
        } finally {
            setExporting(false);
        }
    };

    return (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
            {/* Header / Controls */}
            <div className="p-6 border-b border-gray-100 flex flex-col md:flex-row justify-between items-center gap-4">
                <h3 className="font-bold text-gray-800 text-lg">Tracker Logs</h3>
                
                <div className="flex gap-3 w-full md:w-auto">
                    <div className="relative flex-1 md:w-64">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
                        <input 
                            type="text" 
                            placeholder="Search logs..." 
                            className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                            value={globalFilter ?? ''}
                            onChange={e => setGlobalFilter(e.target.value)}
                        />
                    </div>
                    
                    <button 
                        onClick={exportToExcel}
                        disabled={exporting}
                        className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 transition-colors disabled:opacity-50"
                    >
                        {exporting ? <Loader2 className="animate-spin" size={16} /> : <Download size={16} />}
                        {exporting ? 'Exporting...' : 'Export All'}
                    </button>
                </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
                <table className="w-full">
                    <thead className="bg-gray-50 text-left">
                        {table.getHeaderGroups().map(headerGroup => (
                            <tr key={headerGroup.id}>
                                {headerGroup.headers.map(header => (
                                    <th key={header.id} className="px-6 py-3 text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        {flexRender(header.column.columnDef.header, header.getContext())}
                                    </th>
                                ))}
                            </tr>
                        ))}
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                        {loading ? (
                            <tr>
                                <td colSpan={columns.length} className="px-6 py-8 text-center text-gray-500">
                                    <div className="flex justify-center items-center gap-2">
                                        <Loader2 className="animate-spin" size={20} />
                                        Loading data...
                                    </div>
                                </td>
                            </tr>
                        ) : data.length === 0 ? (
                            <tr>
                                <td colSpan={columns.length} className="px-6 py-8 text-center text-gray-500">
                                    No records found.
                                </td>
                            </tr>
                        ) : (
                            table.getRowModel().rows.map(row => (
                                <tr key={row.id} className="hover:bg-gray-50 transition-colors">
                                    {row.getVisibleCells().map(cell => (
                                        <td key={cell.id} className="px-6 py-4 text-sm text-gray-600 whitespace-nowrap">
                                            {flexRender(cell.column.columnDef.cell, cell.getContext())}
                                        </td>
                                    ))}
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>

            {/* Pagination */}
            <div className="px-6 py-4 border-t border-gray-100 flex flex-col sm:flex-row justify-between items-center gap-4 bg-gray-50">
                <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-500">
                        Page {table.getState().pagination.pageIndex + 1} of {table.getPageCount()}
                    </span>
                    <select
                        value={table.getState().pagination.pageSize}
                        onChange={e => {
                            table.setPageSize(Number(e.target.value));
                        }}
                        className="text-sm border-gray-200 rounded p-1 bg-white text-gray-600 focus:ring-blue-500 border"
                    >
                        {[10, 20, 50, 100].map(pageSize => (
                            <option key={pageSize} value={pageSize}>
                                Show {pageSize}
                            </option>
                        ))}
                    </select>
                </div>
                
                <span className="text-sm text-gray-500 hidden sm:block">
                    Showing {table.getState().pagination.pageIndex * table.getState().pagination.pageSize + 1} to {Math.min((table.getState().pagination.pageIndex + 1) * table.getState().pagination.pageSize, totalRows)} of {totalRows} entries
                </span>
                
                <div className="flex gap-2">
                    <button
                        className="p-2 rounded hover:bg-gray-200 disabled:opacity-50 transition-colors"
                        onClick={() => table.previousPage()}
                        disabled={!table.getCanPreviousPage()}
                    >
                        <ChevronLeft size={16} />
                    </button>
                    <button
                        className="p-2 rounded hover:bg-gray-200 disabled:opacity-50 transition-colors"
                        onClick={() => table.nextPage()}
                        disabled={!table.getCanNextPage()}
                    >
                        <ChevronRight size={16} />
                    </button>
                </div>
            </div>
        </div>
    );
}
