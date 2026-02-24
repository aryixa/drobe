import React, { useState } from 'react'
import { WardrobeGrid } from '../components/wardrobe/WardrobeGrid'
import { UploadModal } from '../components/ui/UploadModal'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { useWardrobe } from '../hooks/useWardrobe'
import { supabase, WardrobeItemInsert, Category } from '../lib/supabase'
import { 
  Search, 
  Filter, 
  Plus, 
  Grid3X3,
  X
} from 'lucide-react'

export function Wardrobe() {
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<Category>('all')
  const [showFilters, setShowFilters] = useState(false)
  
  const { items, loading, error, fetchWardrobe, searchItems, deleteItem } = useWardrobe()

  const categories: Category[] = ['all', 'top', 'bottom', 'shoes', 'accessory']

  const handleSearch = (query: string) => {
    setSearchQuery(query)
    if (query.trim()) {
      searchItems(query, selectedCategory)
    } else {
      fetchWardrobe(selectedCategory)
    }
  }

  const handleCategoryChange = (category: Category) => {
    setSelectedCategory(category)
    if (searchQuery.trim()) {
      searchItems(searchQuery, category)
    } else {
      fetchWardrobe(category)
    }
  }

  const handleUpload = async (item: Omit<WardrobeItemInsert, 'user_id'>) => {
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) throw new Error('User not authenticated')

    const { error } = await supabase
      .from('wardrobe_items')
      .insert({
        ...item,
        user_id: user.id
      })

    if (error) throw error

    // Refresh the wardrobe
    fetchWardrobe(selectedCategory)
  }

  const handleDelete = async (itemId: string) => {
    try {
      await deleteItem(itemId)
    } catch (err) {
      console.error('Failed to delete item:', err)
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-primary-900 mb-2">My Wardrobe</h1>
          <p className="text-primary-600">
            {items.length} {items.length === 1 ? 'item' : 'items'} in your collection
          </p>
        </div>
        <Button
          onClick={() => setIsUploadModalOpen(true)}
          className="mt-4 sm:mt-0"
        >
          <Plus className="mr-2 h-4 w-4" />
          Add Item
        </Button>
      </div>

      {/* Search and Filters */}
      <div className="mb-6 space-y-4">
        {/* Search Bar */}
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-5 w-5 text-primary-400" />
          </div>
          <Input
            placeholder="Search your wardrobe..."
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            className="pl-10"
          />
        </div>

        {/* Filter Controls */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-3 sm:space-y-0">
          <div className="flex items-center space-x-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center"
            >
              <Filter className="mr-2 h-4 w-4" />
              Filters
              {selectedCategory !== 'all' && (
                <span className="ml-2 bg-primary-200 text-primary-800 px-2 py-0.5 rounded-full text-xs">
                  1
                </span>
              )}
            </Button>
          </div>

          {selectedCategory !== 'all' && (
            <div className="flex items-center space-x-2">
              <span className="text-sm text-primary-600">Filtered by:</span>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800">
                {selectedCategory}
                <button
                  onClick={() => handleCategoryChange('all')}
                  className="ml-1.5 text-primary-600 hover:text-primary-800"
                >
                  <X className="h-3 w-3" />
                </button>
              </span>
            </div>
          )}
        </div>

        {/* Filter Panel */}
        {showFilters && (
          <div className="card p-4">
            <h3 className="text-sm font-medium text-primary-900 mb-3">Categories</h3>
            <div className="flex flex-wrap gap-2">
              {categories.map((category) => (
                <button
                  key={category}
                  onClick={() => handleCategoryChange(category)}
                  className={`
                    px-3 py-1.5 text-sm font-medium rounded-md transition-colors
                    ${selectedCategory === category
                      ? 'bg-primary-900 text-white'
                      : 'bg-primary-100 text-primary-700 hover:bg-primary-200'
                    }
                  `}
                >
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                  {category !== 'all' && (
                    <span className="ml-1.5 text-xs opacity-75">
                      ({items.filter(item => item.category === category).length})
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Error State */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-md">
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}

      {/* Wardrobe Grid */}
      <WardrobeGrid
        items={items}
        loading={loading}
        onDelete={handleDelete}
      />

      {/* Upload Modal */}
      <UploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onUpload={handleUpload}
      />
    </div>
  )
}
