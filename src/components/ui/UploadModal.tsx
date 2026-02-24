import React, { useState, useRef } from 'react'
import { Button } from './Button'
import { Input } from './Input'
import { Category, WardrobeItem } from '../../lib/supabase'
import { uploadImage, createImagePreview, validateFile } from '../../lib/storage'
import { X, Upload, Image as ImageIcon } from 'lucide-react'

interface UploadModalProps {
  isOpen: boolean
  onClose: () => void
  onUpload: (item: Omit<WardrobeItem, 'id' | 'user_id' | 'created_at'>) => Promise<void>
}

export function UploadModal({ isOpen, onClose, onUpload }: UploadModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string>('')
  const [name, setName] = useState('')
  const [category, setCategory] = useState<'top' | 'bottom' | 'shoes' | 'accessory'>('top')
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const categories: ('top' | 'bottom' | 'shoes' | 'accessory')[] = ['top', 'bottom', 'shoes', 'accessory']

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const validation = validateFile(file)
    if (!validation.isValid) {
      setError(validation.error || 'Invalid file')
      return
    }

    setSelectedFile(file)
    setError('')

    // Create preview
    createImagePreview(file)
      .then(setPreview)
      .catch(err => {
        console.error('Failed to create preview:', err)
        setError('Failed to create image preview')
      })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!selectedFile || !name.trim()) {
      setError('Please select a file and enter a name')
      return
    }

    setLoading(true)
    setError('')
    setProgress(0)

    try {
      // Upload image
      const { url, path } = await uploadImage(
        selectedFile,
        'temp-user-id', // This will be replaced with actual user ID
        (progress) => setProgress(progress)
      )

      // Create wardrobe item
      await onUpload({
        name: name.trim(),
        category,
        image_url: url,
        image_path: path
      })

      // Reset form
      setSelectedFile(null)
      setPreview('')
      setName('')
      setCategory('top' as 'top' | 'bottom' | 'shoes' | 'accessory')
      setProgress(0)
      onClose()
    } catch (err) {
      console.error('Upload failed:', err)
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setLoading(false)
    }
  }

  const handleClose = () => {
    if (!loading) {
      setSelectedFile(null)
      setPreview('')
      setName('')
      setCategory('top' as 'top' | 'bottom' | 'shoes' | 'accessory')
      setProgress(0)
      setError('')
      onClose()
    }
  }

  if (!isOpen) return null

  return (
    <div className="modal-overlay">
      <div className="modal-content animate-in">
        <div className="flex items-center justify-between p-6 border-b border-primary-200">
          <h2 className="text-lg font-semibold text-primary-900">Add to Wardrobe</h2>
          <button
            onClick={handleClose}
            disabled={loading}
            className="text-primary-400 hover:text-primary-600 disabled:opacity-50"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6">
          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
              <p className="text-sm text-red-600">{error}</p>
            </div>
          )}

          {/* File Upload Area */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-primary-700 mb-2">
              Image
            </label>
            <div
              onClick={() => fileInputRef.current?.click()}
              className={`
                border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
                ${preview 
                  ? 'border-primary-300 bg-primary-50' 
                  : 'border-primary-300 hover:border-primary-400 hover:bg-primary-50'
                }
              `}
            >
              {preview ? (
                <div className="space-y-3">
                  <img
                    src={preview}
                    alt="Preview"
                    className="mx-auto max-h-32 rounded-md object-cover"
                  />
                  <p className="text-sm text-primary-600">{selectedFile?.name}</p>
                </div>
              ) : (
                <div className="space-y-2">
                  <ImageIcon className="mx-auto h-12 w-12 text-primary-400" />
                  <div>
                    <p className="text-sm text-primary-600">Click to upload image</p>
                    <p className="text-xs text-primary-400">JPEG, PNG up to 15MB</p>
                  </div>
                </div>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/jpg,image/png"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>

          {/* Name Input */}
          <div className="mb-4">
            <Input
              label="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Summer T-Shirt"
              disabled={loading}
            />
          </div>

          {/* Category Select */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-primary-700 mb-2">
              Category
            </label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value as 'top' | 'bottom' | 'shoes' | 'accessory')}
              disabled={loading}
              className="w-full px-3 py-2 border border-primary-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
            >
              {categories.map((cat) => (
                <option key={cat} value={cat}>
                  {cat.charAt(0).toUpperCase() + cat.slice(1)}
                </option>
              ))}
            </select>
          </div>

          {/* Progress Bar */}
          {loading && progress > 0 && (
            <div className="mb-4">
              <div className="flex justify-between text-sm text-primary-600 mb-1">
                <span>Uploading...</span>
                <span>{progress}%</span>
              </div>
              <div className="w-full bg-primary-200 rounded-full h-2">
                <div
                  className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex space-x-3">
            <Button
              type="submit"
              loading={loading}
              progress={loading ? progress : undefined}
              disabled={!selectedFile || !name.trim()}
              className="flex-1"
            >
              {loading ? 'Uploading...' : 'Add to Wardrobe'}
            </Button>
            <Button
              type="button"
              variant="secondary"
              onClick={handleClose}
              disabled={loading}
            >
              Cancel
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
