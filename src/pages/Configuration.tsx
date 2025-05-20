import { useState } from 'react'

export default function Configuration() {
  const [config, setConfig] = useState({
    apiKey: '',
    apiSecret: '',
    maxPositions: 3,
    riskLevel: 'medium',
    tradingPairs: ['BTC/USD'],
  })

  const saveConfig = async () => {
    try {
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      // Show success message
    } catch (error) {
      // Show error message
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Configuration</h1>

      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <form onSubmit={(e) => { e.preventDefault(); saveConfig(); }} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              API Key
            </label>
            <input
              type="password"
              value={config.apiKey}
              onChange={(e) => setConfig({ ...config, apiKey: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              API Secret
            </label>
            <input
              type="password"
              value={config.apiSecret}
              onChange={(e) => setConfig({ ...config, apiSecret: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Max Positions
            </label>
            <input
              type="number"
              value={config.maxPositions}
              onChange={(e) => setConfig({ ...config, maxPositions: parseInt(e.target.value) })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Risk Level
            </label>
            <select
              value={config.riskLevel}
              onChange={(e) => setConfig({ ...config, riskLevel: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Trading Pairs
            </label>
            <select
              multiple
              value={config.tradingPairs}
              onChange={(e) => setConfig({ ...config, tradingPairs: Array.from(e.target.selectedOptions, option => option.value) })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600"
            >
              <option value="BTC/USD">BTC/USD</option>
              <option value="ETH/USD">ETH/USD</option>
              <option value="SOL/USD">SOL/USD</option>
            </select>
          </div>

          <div className="flex justify-end">
            <button
              type="submit"
              className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Save Configuration
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}