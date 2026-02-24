import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useWardrobe } from '../hooks/useWardrobe'
import { Button } from '../components/ui/Button'
import { 
  Plus, 
  Grid3X3, 
  TrendingUp, 
  Calendar,
  Shirt,
  ShoppingBag,
  Heart
} from 'lucide-react'

export function Dashboard() {
  const { user } = useAuth()
  const { items } = useWardrobe()
  const navigate = useNavigate()

  const categories = [
    { name: 'top', icon: Shirt, count: items.filter(item => item.category === 'top').length },
    { name: 'bottom', icon: ShoppingBag, count: items.filter(item => item.category === 'bottom').length },
    { name: 'shoes', icon: TrendingUp, count: items.filter(item => item.category === 'shoes').length },
    { name: 'accessory', icon: Heart, count: items.filter(item => item.category === 'accessory').length },
  ]

  const recentItems = items.slice(0, 3)

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Welcome Section */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-primary-900 mb-2">
          Welcome back, {user?.email?.split('@')[0] || 'User'}!
        </h1>
        <p className="text-primary-600">
          Here's what's happening with your wardrobe today.
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Grid3X3 className="h-8 w-8 text-primary-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-primary-600">Total Items</p>
              <p className="text-2xl font-semibold text-primary-900">{items.length}</p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Calendar className="h-8 w-8 text-primary-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-primary-600">This Month</p>
              <p className="text-2xl font-semibold text-primary-900">
                {items.filter(item => {
                  const itemDate = new Date(item.created_at)
                  const now = new Date()
                  return itemDate.getMonth() === now.getMonth() && 
                         itemDate.getFullYear() === now.getFullYear()
                }).length}
              </p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <TrendingUp className="h-8 w-8 text-accent-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-primary-600">Most Worn</p>
              <p className="text-2xl font-semibold text-primary-900">Tops</p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Plus className="h-8 w-8 text-primary-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-primary-600">Quick Add</p>
              <Button
                size="sm"
                onClick={() => navigate('/wardrobe')}
                className="mt-1"
              >
                Add Item
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Categories Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-primary-900 mb-4">Categories</h2>
          <div className="space-y-3">
            {categories.map((category) => {
              const Icon = category.icon
              return (
                <div
                  key={category.name}
                  className="flex items-center justify-between p-3 bg-primary-50 rounded-lg hover:bg-primary-100 transition-colors cursor-pointer"
                  onClick={() => navigate('/wardrobe')}
                >
                  <div className="flex items-center space-x-3">
                    <Icon className="h-5 w-5 text-primary-600" />
                    <span className="font-medium text-primary-900 capitalize">
                      {category.name}s
                    </span>
                  </div>
                  <span className="text-sm text-primary-600 bg-primary-200 px-2 py-1 rounded-full">
                    {category.count}
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Recent Items */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-primary-900">Recent Items</h2>
            <Button variant="ghost" size="sm" onClick={() => navigate('/wardrobe')}>
              View All
            </Button>
          </div>
          <div className="space-y-3">
            {recentItems.length > 0 ? (
              recentItems.map((item) => (
                <div
                  key={item.id}
                  className="flex items-center space-x-3 p-3 bg-primary-50 rounded-lg hover:bg-primary-100 transition-colors cursor-pointer"
                  onClick={() => navigate('/wardrobe')}
                >
                  <img
                    src={item.image_url}
                    alt={item.name}
                    className="w-12 h-12 object-cover rounded-md"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-primary-900 truncate">{item.name}</p>
                    <p className="text-sm text-primary-600 capitalize">{item.category}</p>
                  </div>
                  <span className="text-xs text-primary-500">
                    {new Date(item.created_at).toLocaleDateString()}
                  </span>
                </div>
              ))
            ) : (
              <div className="text-center py-8">
                <div className="mx-auto h-12 w-12 text-primary-400 mb-3">
                  <Grid3X3 />
                </div>
                <p className="text-primary-600 mb-3">No items yet</p>
                <Button onClick={() => navigate('/wardrobe')}>
                  Add Your First Item
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card p-6">
        <h2 className="text-lg font-semibold text-primary-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Button
            onClick={() => navigate('/wardrobe')}
            className="flex items-center justify-center"
          >
            <Plus className="mr-2 h-4 w-4" />
            Add New Item
          </Button>
          <Button
            variant="secondary"
            onClick={() => navigate('/wardrobe')}
            className="flex items-center justify-center"
          >
            <Grid3X3 className="mr-2 h-4 w-4" />
            Browse Wardrobe
          </Button>
          <Button
            variant="secondary"
            onClick={() => navigate('/stylist')}
            className="flex items-center justify-center"
          >
            <Heart className="mr-2 h-4 w-4" />
            Get Styling Tips
          </Button>
        </div>
      </div>
    </div>
  )
}
