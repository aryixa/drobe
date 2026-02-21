import { useState, useEffect } from 'react'
import { supabase, WardrobeItem, Category } from '../lib/supabase'
import { useAuth } from '../context/AuthContext'

export function useWardrobe() {
  const { user } = useAuth()
  const [items, setItems] = useState<WardrobeItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch wardrobe items
  const fetchWardrobe = async (category?: Category) => {
    if (!user) return

    setLoading(true)
    setError(null)

    try {
      let query = supabase
        .from('wardrobe_items')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })
        .limit(50)

      if (category && category !== 'all') {
        query = query.eq('category', category)
      }

      const { data, error } = await query

      if (error) throw error
      setItems(data || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch wardrobe items')
    } finally {
      setLoading(false)
    }
  }

  // Search items
  const searchItems = async (query: string, category?: Category) => {
    if (!user) return

    setLoading(true)
    setError(null)

    try {
      let dbQuery = supabase
        .from('wardrobe_items')
        .select('*')
        .eq('user_id', user.id)
        .ilike('name', `%${query}%`)
        .order('created_at', { ascending: false })
        .limit(50)

      if (category && category !== 'all') {
        dbQuery = dbQuery.eq('category', category)
      }

      const { data, error } = await dbQuery

      if (error) throw error
      setItems(data || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to search items')
    } finally {
      setLoading(false)
    }
  }

  // Delete item with optimistic UI
  const deleteItem = async (itemId: string) => {
    if (!user) return

    const originalItems = [...items]
    
    // Optimistic removal
    setItems(items.filter(item => item.id !== itemId))

    try {
      const { error } = await supabase
        .from('wardrobe_items')
        .delete()
        .eq('id', itemId)
        .eq('user_id', user.id)

      if (error) throw error
    } catch (err) {
      // Revert on error
      setItems(originalItems)
      setError(err instanceof Error ? err.message : 'Failed to delete item')
      throw err
    }
  }

  // Initial fetch
  useEffect(() => {
    if (user) {
      fetchWardrobe()
    } else {
      setItems([])
    }
  }, [user])

  return {
    items,
    loading,
    error,
    fetchWardrobe,
    searchItems,
    deleteItem,
  }
}
