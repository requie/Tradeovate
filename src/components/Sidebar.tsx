import { Link } from 'react-router-dom'
import {
  HomeIcon,
  Cog6ToothIcon,
  ChartBarIcon,
  WrenchScrewdriverIcon,
} from '@heroicons/react/24/outline'

interface SidebarProps {
  open: boolean
  setOpen: (open: boolean) => void
}

export default function Sidebar({ open, setOpen }: SidebarProps) {
  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
    { name: 'Configuration', href: '/configuration', icon: WrenchScrewdriverIcon },
    { name: 'Monitoring', href: '/monitoring', icon: ChartBarIcon },
    { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
  ]

  return (
    <div className={`${open ? 'block' : 'hidden'} lg:block lg:w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700`}>
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-center h-16 border-b border-gray-200 dark:border-gray-700">
          <h1 className="text-xl font-bold text-gray-800 dark:text-white">Trading Bot</h1>
        </div>

        <nav className="flex-1 overflow-y-auto">
          <ul className="p-4 space-y-2">
            {navigation.map((item) => (
              <li key={item.name}>
                <Link
                  to={item.href}
                  className="flex items-center p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                >
                  <item.icon className="w-6 h-6 mr-3" />
                  {item.name}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </div>
  )
}