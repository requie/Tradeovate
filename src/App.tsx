import { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import Configuration from './pages/Configuration'
import Monitoring from './pages/Monitoring'
import Settings from './pages/Settings'
import { ThemeProvider } from './contexts/ThemeContext'

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <BrowserRouter>
      <ThemeProvider>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route
            path="/*"
            element={
              <div className="flex h-screen bg-gray-100 dark:bg-gray-900">
                <Sidebar open={sidebarOpen} setOpen={setSidebarOpen} />
                <div className="flex-1 flex flex-col overflow-hidden">
                  <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-100 dark:bg-gray-900">
                    <div className="container mx-auto px-6 py-8">
                      <Routes>
                        <Route path="/dashboard" element={<Dashboard />} />
                        <Route path="/configuration" element={<Configuration />} />
                        <Route path="/monitoring" element={<Monitoring />} />
                        <Route path="/settings" element={<Settings />} />
                      </Routes>
                    </div>
                  </main>
                </div>
              </div>
            }
          />
        </Routes>
      </ThemeProvider>
    </BrowserRouter>
  )
}