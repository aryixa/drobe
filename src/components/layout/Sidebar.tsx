import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { clsx } from 'clsx'
import { 
  Home, 
  Grid3X3, 
  MessageSquare, 
  User, 
  Settings,
  LogOut,
  Plus
} from 'lucide-react'

interface SidebarProps {
  onUploadClick: () => void
  onSignOut: () => void
}

export function Sidebar({ onUploadClick, onSignOut }: SidebarProps) {
  const location = useLocation()

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: Home },
    { name: 'Wardrobe', href: '/wardrobe', icon: Grid3X3 },
    { name: 'Stylist', href: '/stylist', icon: MessageSquare },
  ]

  const isActive = (path: string) => {
    return location.pathname === path
  }

  return (
    <div className="flex flex-col h-full bg-white border-r border-primary-200">
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-primary-200">
        <div>
          <h1 className="text-xl font-bold text-primary-900">Wardrobe Vault</h1>
          <p className="text-xs text-primary-500">Minimalist Wardrobe Management</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-1">
        {navigation.map((item) => {
          const Icon = item.icon
          return (
            <NavLink
              key={item.name}
              to={item.href}
              className={clsx(
                'flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors duration-200',
                isActive(item.href)
                  ? 'bg-primary-100 text-primary-900'
                  : 'text-primary-600 hover:bg-primary-50 hover:text-primary-900'
              )}
            >
              <Icon className="mr-3 h-5 w-5" />
              {item.name}
            </NavLink>
          )
        })}
      </nav>

      {/* Actions */}
      <div className="p-4 border-t border-primary-200 space-y-2">
        <button
          onClick={onUploadClick}
          className="w-full flex items-center justify-center px-3 py-2 text-sm font-medium text-white bg-primary-900 rounded-md hover:bg-primary-800 transition-colors duration-200"
        >
          <Plus className="mr-2 h-4 w-4" />
          Add Item
        </button>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-primary-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-primary-200 rounded-full flex items-center justify-center">
              <User className="h-4 w-4 text-primary-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-primary-900">User</p>
              <p className="text-xs text-primary-500">user@example.com</p>
            </div>
          </div>
          <button
            onClick={onSignOut}
            className="text-primary-400 hover:text-primary-600 transition-colors duration-200"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  )
}
