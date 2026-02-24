import React from 'react'
import { WardrobeItem } from '../../lib/supabase'
import { Button } from '../ui/Button'
import { X, Search, Grid, List } from 'lucide-react'

interface WardrobeGridProps {
  items: WardrobeItem[]
  loading: boolean
  onDelete: (id: string) => void
}

export function WardrobeGrid({ items, loading, onDelete }: WardrobeGridProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {[...Array(8)].map((_, i) => (
          <div key={i} className="aspect-square bg-primary-100 rounded-lg animate-pulse" />
        ))}
      </div>
    )
  }

  if (items.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="mx-auto max-w-sm">
          <div className="mx-auto h-12 w-12 text-primary-400 mb-4">
            <Grid />
          </div>
          <h3 className="text-lg font-medium text-primary-900 mb-2">No items yet</h3>
          <p className="text-primary-600 mb-4">Start building your wardrobe by adding your first item.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
      {items.map((item) => (
        <div key={item.id} className="group relative aspect-square card overflow-hidden hover-lift">
          <img
            src={item.image_url}
            alt={item.name}
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-40 transition-opacity duration-200">
            <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onDelete(item.id)}
                className="bg-white/90 hover:bg-white text-red-600 hover:text-red-700"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200">
              <h4 className="text-white font-medium text-sm truncate">{item.name}</h4>
              <p className="text-white/80 text-xs capitalize">{item.category}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
