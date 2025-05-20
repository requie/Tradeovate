import { useState } from 'react'
import { useTheme } from '../contexts/ThemeContext'

export default function Settings() {
  const { isDark, toggleTheme } = useTheme()
  const [settings, setSettings] = useState({
    notifications: true,
    autoRestart: false,
    logLevel: 'info',
    backupFrequency: 'daily'
  })

  const saveSettings = async () => {
    try {
      await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      })
      // Show success message
    } catch (error) {
      // Show error message
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Settings</h1>

      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
        <form onSubmit={(e) => { e.preventDefault(); saveSettings(); }} className="space-y-6">
          {/* Theme Toggle */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Dark Mode</span>
            <button
              type="button"
              onClick={toggleTheme}
              className={`${
                isDark ? 'bg-indigo-600' : 'bg-gray-200'
              } relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2`}
            >
              <span
                className={`${
                  isDark ? 'translate-x-5' : 'translate-x-0'
                } pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
              />
            </button>
          </div>

          {/* Notifications */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Notifications</span>
            <button
              type="button"
              onClick={() => setSettings({ ...settings, notifications: !settings.notifications })}
              className={`${
                settings.notifications ? 'bg-indigo-600' : 'bg-gray-200'
              } relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2`}
            >
              <span
                className={`${
                  settings.notifications ? 'translate-x-5' : 'translate-x-0'
                } pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
              />
            </button>
          </div>

          {/* Auto Restart */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Auto Restart</span>
            <button
              type="button"
              onClick={() => setSettings({ ...settings, autoRestart: !settings.autoRestart })}
              className={`${
                settings.autoRestart ? 'bg-indigo-600' : 'bg-gray-200'
              } relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2`}
            >
              <span
                className={`${
                  settings.autoRestart ? 'translate-x-5' : 'translate-x-0'
                } pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
              />
            </button>
          </div>

          {/* Log Level */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Log Level
            </label>
            <select
              value={settings.logLevel}
              onChange={(e) => setSettings({ ...settings, logLevel: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600"
            >
              <option value="debug">Debug</option>
              <option value="info">Info</option>
              <option value="warning">Warning</option>
              <option value="error">Error</option>
            </select>
          </div>

          {/* Backup Frequency */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Backup Frequency
            </label>
            <select
              value={settings.backupFrequency}
              onChange={(e) => setSettings({ ...settings, backupFrequency: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600"
            >
              <option value="hourly">Hourly</option>
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
            </select>
          </div>

          <div className="flex justify-end">
            <button
              type="submit"
              className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Save Settings
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}