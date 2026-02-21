import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables. Please check your .env file.')
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

export type Database = {
  public: {
    Tables: {
      wardrobe_items: {
        Row: {
          id: string
          user_id: string
          name: string
          category: 'top' | 'bottom' | 'shoes' | 'accessory'
          image_url: string
          image_path: string
          created_at: string
        }
        Insert: {
          id?: string
          user_id: string
          name: string
          category?: 'top' | 'bottom' | 'shoes' | 'accessory'
          image_url: string
          image_path: string
          created_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          name?: string
          category?: 'top' | 'bottom' | 'shoes' | 'accessory'
          image_url?: string
          image_path?: string
          created_at?: string
        }
      }
    }
  }
}

export type WardrobeItem = Database['public']['Tables']['wardrobe_items']['Row']
export type WardrobeItemInsert = Database['public']['Tables']['wardrobe_items']['Insert']
export type WardrobeItemUpdate = Database['public']['Tables']['wardrobe_items']['Update']

export type Category = 'top' | 'bottom' | 'shoes' | 'accessory' | 'all'

// Vite environment variable types
interface ImportMetaEnv {
  readonly VITE_SUPABASE_URL: string
  readonly VITE_SUPABASE_ANON_KEY: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
