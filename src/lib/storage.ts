import { supabase } from './supabase'

export interface FileValidation {
  isValid: boolean
  error?: string
}

export function validateFile(file: File): FileValidation {
  // Check file size (15MB limit)
  const maxSize = 15 * 1024 * 1024 // 15MB in bytes
  if (file.size > maxSize) {
    return {
      isValid: false,
      error: 'File size must be less than 15MB'
    }
  }

  // Check file type (JPEG/PNG only)
  const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png']
  if (!allowedTypes.includes(file.type)) {
    return {
      isValid: false,
      error: 'Only JPEG and PNG images are allowed'
    }
  }

  return { isValid: true }
}

export function generateStoragePath(userId: string, fileName: string): string {
  const timestamp = Date.now()
  const sanitizedName = fileName.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9._-]/g, '')
  return `${userId}/${timestamp}_${sanitizedName}`
}

export function createImagePreview(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      if (e.target?.result) {
        resolve(e.target.result as string)
      } else {
        reject(new Error('Failed to create image preview'))
      }
    }
    reader.onerror = () => reject(new Error('Failed to read file'))
    reader.readAsDataURL(file)
  })
}

export async function uploadImage(
  file: File,
  userId: string,
  onProgress?: (progress: number) => void
): Promise<{ url: string; path: string }> {
  const validation = validateFile(file)
  if (!validation.isValid) {
    throw new Error(validation.error)
  }

  const filePath = generateStoragePath(userId, file.name)
  
  return new Promise((resolve, reject) => {
    const { data, error } = supabase.storage
      .from('wardrobe')
      .upload(filePath, file, {
        cacheControl: '3600',
        upsert: false,
        onUploadProgress: (progress: any) => {
          if (onProgress) {
            const percentage = (progress.loaded / progress.total) * 100
            onProgress(Math.round(percentage))
          }
        }
      })

    if (error) {
      reject(error)
      return
    }

    if (!data?.path) {
      reject(new Error('Upload failed: No path returned'))
      return
    }

    // Get public URL
    const { data: { publicUrl } } = supabase.storage
      .from('wardrobe')
      .getPublicUrl(data.path)

    resolve({
      url: publicUrl,
      path: data.path
    })
  })
}

export async function deleteImage(imagePath: string): Promise<void> {
  const { error } = await supabase.storage
    .from('wardrobe')
    .remove([imagePath])

  if (error) {
    throw new Error(`Failed to delete image from storage: ${error.message}`)
  }
}
