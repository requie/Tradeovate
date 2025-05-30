import { useState, useEffect } from 'react'
import { useQuery } from 'react-query'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts'

export default function Dashboard() {
  const [botStatus, setBotStatus] = useState('stopped')
  const [performance, setPerformance] = useState([])

  const { data: botStats, isLoading } = useQuery('botStats', async () => {
    const response = await fetch('https://localhost:8000/api/stats')
    if (!response.ok) {
      throw new Error('Network response was not ok')
    }
    return response.json()
  })

  const toggleBot = async () => {
    const newStatus = botStatus === 'running' ? 'stopped' : 'running'
    const response = await fetch('https://localhost:8000/api/bot/status', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: newStatus })
    })
    if (response.ok) {
      setBotStatus(newStatus)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
        <button
          onClick={toggleBot}
          className={`px-4 py-2 rounded-lg ${
            botStatus === 'running'
              ? 'bg-red-500 hover:bg-red-600'
              : 'bg-green-500 hover:bg-green-600'
          } text-white`}
        >
          {botStatus === 'running' ? 'Stop Bot' : 'Start Bot'}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Quick Stats */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200">Total Trades</h3>
          <p className="text-3xl font-bold text-gray-900 dark:text-white">
            {isLoading ? '...' : botStats?.totalTrades || 0}
          </p>
        </div>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200">Success Rate</h3>
          <p className="text-3xl font-bold text-gray-900 dark:text-white">
            {isLoading ? '...' : `${botStats?.successRate || 0}%`}
          </p>
        </div>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200">Total Profit</h3>
          <p className="text-3xl font-bold text-gray-900 dark:text-white">
            {isLoading ? '...' : `$${botStats?.totalProfit || 0}`}
          </p>
        </div>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200">Active Positions</h3>
          <p className="text-3xl font-bold text-gray-900 dark:text-white">
            {isLoading ? '...' : botStats?.activePositions || 0}
          </p>
        </div>
      </div>

      {/* Performance Chart */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">Performance</h2>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={performance}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="value" stroke="#4F46E5" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">Recent Activity</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead>
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Time</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Action</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Details</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {/* Activity rows would be dynamically populated */}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}